"""Compile the app's bounded JSON subset to GBNF without native enum escaping.

llama.cpp b7099's JSON-schema converter does not escape backslashes in JSON
string literals a second time. Quotes/backslashes from speech can therefore
invalidate its grammar. Keep the schema and all source strings unchanged;
serialize the JSON value, then quote that serialization for the GBNF literal.
"""
from __future__ import annotations

import json


def literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def compile_schema(schema: dict) -> str:
    """Only the explicit, bounded schema features used by local requests.

    Unsupported/optional object structures fail closed instead of silently
    dropping constraints. Local response validation remains mandatory.
    """
    rules = {}
    memo = {}

    def emit(node):
        if not isinstance(node, dict):
            raise ValueError('Ожидалась схема объекта JSON')
        key = json.dumps(node, ensure_ascii=False, sort_keys=True)
        if key in memo:
            return memo[key]
        name = 'v' + str(len(memo))
        memo[key] = name
        if 'anyOf' in node:
            if set(node) != {'anyOf'} or not isinstance(node['anyOf'], list) or not node['anyOf']:
                raise ValueError('Некорректные варианты схемы JSON')
            body = ' | '.join(emit(branch) for branch in node['anyOf'])
        elif 'enum' in node:
            if set(node) != {'type', 'enum'} or node['type'] not in ('string', 'integer'):
                raise ValueError('Неподдерживаемое перечисление JSON')
            values = node['enum']
            if not isinstance(values, list) or not values:
                raise ValueError('Пустое перечисление JSON')
            for value in values:
                if (node['type'] == 'string' and not isinstance(value, str)
                        or node['type'] == 'integer' and type(value) is not int):
                    raise ValueError('Неверный тип значения перечисления')
            body = ' | '.join(literal(json.dumps(value, ensure_ascii=False)) for value in values)
        elif node.get('type') == 'object':
            if set(node) != {'type', 'properties', 'required', 'additionalProperties'} or node['additionalProperties'] is not False:
                raise ValueError('Объект должен явно запрещать дополнительные поля')
            props, required = node['properties'], node['required']
            if not isinstance(props, dict) or not isinstance(required, list) or len(required) != len(props) or set(required) != set(props):
                raise ValueError('Все поля объекта должны быть обязательными')
            entries = [literal(json.dumps(key, ensure_ascii=False)) + ' ws ":" ws ' + emit(value)
                       for key, value in props.items()]
            body = '"{" ws ' + ' ws "," ws '.join(entries) + ' ws "}"'
        elif node.get('type') == 'array':
            if set(node) - {'type', 'items', 'minItems', 'maxItems'}:
                raise ValueError('Неподдерживаемые ограничения массива')
            lo, hi = node.get('minItems', 0), node.get('maxItems')
            if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= 100:
                raise ValueError('Массив должен иметь конечную допустимую длину')
            item = emit(node['items'])
            if hi == 0:
                content = ''
            else:
                tail = '(ws "," ws ' + item + '){' + str(max(0, lo-1)) + ',' + str(hi-1) + '}'
                content = '(' + item + ' ' + tail + ')' + ('?' if lo == 0 else '') + ' ws '
            body = '"[" ws ' + content + '"]"'
        elif node.get('type') == 'integer':
            if set(node) != {'type', 'minimum', 'maximum'}:
                raise ValueError('Целое число должно иметь явный диапазон')
            lo, hi = node['minimum'], node['maximum']
            if type(lo) is not int or type(hi) is not int or not lo <= hi or hi-lo > 1000:
                raise ValueError('Недопустимый диапазон целых чисел')
            body = ' | '.join(literal(str(value)) for value in range(lo, hi+1))
        elif node.get('type') == 'string':
            if set(node) != {'type', 'maxLength'} or type(node['maxLength']) is not int or not 0 <= node['maxLength'] <= 2000:
                raise ValueError('Строка должна иметь ограничение длины')
            body = r'"\"" jchar{0,' + str(node['maxLength']) + r'} "\""'
        else:
            raise ValueError('Неподдерживаемая схема JSON')
        rules[name] = body
        return name

    root = emit(schema)
    return '\n'.join([
        'root ::= ws ' + root + ' ws',
        r'ws ::= [ \t\n\r]{0,8}',
        r'jchar ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})',
        *(name + ' ::= ' + body for name, body in rules.items()),
    ]) + '\n'
