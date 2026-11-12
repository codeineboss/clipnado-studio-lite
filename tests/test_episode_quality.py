"""Small hand-authored ranking regressions, not a benchmark on real streams."""
from clipnado.episodes import select_episodes
from clipnado.models import MediaInfo, SearchSettings, Segment
from clipnado.process import Cancellation


def analyze(segments, duration=90000, energy=None):
    media = MediaInfo("fixture.mp4", "fixture-id", duration, 1920, 1080, 60, True)
    return select_episodes(media, segments, SearchSettings(), energy or [], Cancellation())


def conversation(offset=0):
    return [Segment(offset + a, offset + b, text) for a, b, text in [
        (0, 20000, "Почему ты принёс учителю коробку вместо тетради? Мы же договаривались принести выполненное задание."),
        (20000, 45000, "Я хотел сначала открыть коробку и показать, что в ней находится. Все внимательно ждали объяснения."),
        (45000, 60000, "Но тут я открыл крышку, и оттуда вдруг выпрыгнула игрушечная лягушка прямо на стол."),
        (60000, 75000, "Ты серьёзно? Ахаха, я даже подскочил от неожиданности! Ребята рассмеялись и не могли успокоиться."),
        (75000, 80000, "В итоге учитель тоже рассмеялся, убрал лягушку и попросил всё-таки показать тетрадь."),
        (80000, 90000, "Теперь открываем учебник на следующей странице и спокойно переходим к новому заданию.")]]


def test_generic_connectors_and_mixed_words_do_not_make_an_event():
    segments = [Segment(i, i + 10000, "Потом изменим смещение и сменим переменную. Потому что это смешанный формат данных. В итоге всё получилось.")
                for i in range(0, 120000, 10000)]
    assert analyze(segments, 120000, [(45000, .9)]) == []


def test_episode_keeps_setup_and_late_response_instead_of_shortest_boundary():
    clips = analyze(conversation())
    assert clips
    assert any(c.start_ms == 0 and c.end_ms >= 75000 for c in clips)
    assert all(60000 <= c.end_ms - c.start_ms <= 90000 for c in clips)
    assert all(c.provider == "local-context" and c.version == 2 for c in clips)


def test_dialogue_and_laughter_without_story_keywords_are_found():
    segments = [
        Segment(0, 25000, "Зачем ты нажал на эту кнопку? Я просил подождать, пока мы закончим подготовку и проверим подключение."),
        Segment(25000, 40000, "Что ты творишь? Я не ожидал, что из колонки сейчас заговорит мой собственный записанный голос!"),
        Segment(40000, 55000, "Ахаха, все рассмеялись, а я ещё несколько секунд пытался понять, кто со мной разговаривает."),
        Segment(55000, 70000, "Мы остановили запись и разобрались, как случайно включилась эта кнопка воспроизведения голоса.")]
    assert analyze(segments, 70000)


def test_same_event_text_is_not_repeated_in_two_distant_clips():
    clips = analyze(conversation() + conversation(180000), 270000)
    assert len(clips) == 1


def test_event_at_end_of_source_has_valid_context_and_end():
    clips = analyze(conversation(30000), 120000)
    assert clips
    assert any(c.start_ms <= 30000 and c.end_ms >= 105000 for c in clips)
    assert all(c.end_ms <= 120000 for c in clips)


def test_many_distinct_events_can_reach_default_target_without_repeating_windows():
    # Hand-authored scenarios; this tests retrieval/packing, not human interest.
    scenarios = [
        "Учитель открыл коробку вместо тетради. Изнутри выскочила игрушечная лягушка прямо на стол между учебниками.",
        "Повар перепутал соль с сахаром. Праздничный торт оказался солёным пирогом, гости попросили огурцы вместо чая.",
        "Рыбак вытащил тяжёлый сапог. Внутри обнаружилась связка потерянных ключей от соседского гаража возле реки.",
        "Кот включил микрофон лапой. Вся рабочая конференция слушала громкое мурчание, директор попросил нового коллегу представиться.",
        "Фокусник достал морковку из шляпы. Кролик тем временем спокойно сидел в зрительном зале рядом с билетером.",
        "Вратарь побежал праздновать победу. Мяч отскочил от штанги, прокатился назад через ворота, судья отменил чемпионство.",
        "Турист заказал такси на пляж. Навигатор привёз пассажира к песочнице детского сада посреди промышленного района.",
        "Музыкант сел играть сонату. Вместо рояля зазвонил спрятанный телефон, оркестр подхватил мелодию знакомого будильника.",
        "Собака принесла почтальону пакет. Хозяин узнал собственные тапочки, которые искал целую неделю под диваном спальни.",
        "Робот пылесос сбежал из квартиры. Сосед нашёл беглеца возле лифта, украшенного гирляндой после домашнего праздника.",
        "Строитель покрасил дверь зелёным. Заказчик открыл другой подъезд, рабочие обнаружили ошибочный адрес на чертеже ремонта.",
        "Ведущий объявил призовой конкурс. Зритель ответил песней вместо числа, жюри вручило ему неожиданную музыкальную награду.",
    ]
    segments = []
    for index, text in enumerate(scenarios):
        start = index * 90000
        segments.extend([
            Segment(start, start + 20000, "Почему все так удивились? " + text),
            Segment(start + 20000, start + 45000, "Но тут произошло неожиданное. " + text),
            Segment(start + 45000, start + 65000, "Ты серьёзно? Ахаха! " + text),
            Segment(start + 65000, start + 75000, "В итоге все рассмеялись. " + text),
            Segment(start + 75000, start + 90000, "Теперь спокойно переходим к следующей теме и проверяем подключение звука."),
        ])
    clips = analyze(segments, len(scenarios) * 90000)
    assert len(clips) >= 10
    assert all(c.score >= 60 for c in clips)
    assert all(a.end_ms <= b.start_ms for a, b in zip(clips, clips[1:]))
