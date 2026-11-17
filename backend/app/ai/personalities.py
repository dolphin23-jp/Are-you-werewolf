"""Personality system: 15 fixed presets built from 4 orthogonal trait axes
(tone x thinking-style x discussion-style x emotional-tendency), each
rendered as natural-language prompt text. Assigned via seeded RNG so games
are reproducible with a seed, and each preset carries a canned fallback
line matching its tone for total-LLM-failure cases."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

# The structured register (legacy, v2): per-speaker ranges those prompts were
# measured with. The v3 chat register does not use them; see below.
DISCUSSION_LENGTH_RANGES: dict[str, tuple[int, int]] = {
    "terse": (30, 100),
    "normal": (80, 240),
    "wordy": (180, 400),
}

# The chat register (v3). The first Claude game asked normal speakers for
# 80-100 characters and wordy ones for 180-200, and got a table mean of 130:
# the model overshoots whatever it is told by about 1.25-1.6x. So every
# persona gets the same short band for an ordinary turn -- a persona's cadence
# is said in words (`chat_cadence_text`), not as a longer allowance -- and only the
# stages that carry a CO, a result or the day's closing line may run longer.
CHAT_TURN_CHARS: tuple[int, int] = (15, 60)
CHAT_STAGE_MAX_CHARS: dict[str, int] = {
    "reaction": 30,
    "initial_view": 60,
    "rebuttal_or_reassessment": 60,
    "freemason_confirmation": 60,
    "human_followup": 60,
    # From day 2, a seat with no line yet, once, before the summary.
    "open_floor": 60,
    # A seat the summary named, answering it: never the reaction's 30.
    "summary_answer": 60,
    # The facilitator's candidate when the summary named nobody.
    "provisional_designation": 60,
    # The facilitator, after the seats it asked by name have answered.
    "facilitator_followup": 60,
    "immediate": 120,
    "consensus_summary": 120,
    # The close after the summary's answers: today's candidate, in a word.
    "final_designation": 60,
}
# The hard cap: what the agent truncates to and budgets tokens for. Above the
# longest stage so an overshoot is not cut mid-thought; the code-written fact
# sentences are added after the cut, so a result can never be truncated away.
CHAT_MESSAGE_MAX_CHARS = 160


def discussion_length_range(verbosity: str, *, chat: bool = False) -> tuple[int, int]:
    """(minimum, hard cap) for one discussion message.

    `chat` selects the v3 register, where the band is the same for every
    persona; otherwise the per-verbosity range the structured prompts use."""
    if chat:
        return CHAT_TURN_CHARS[0], CHAT_MESSAGE_MAX_CHARS
    return DISCUSSION_LENGTH_RANGES.get(verbosity, DISCUSSION_LENGTH_RANGES["normal"])


def chat_stage_max_chars(stage: str) -> int:
    """The character cap the v3 prompt names for a discussion stage. A stage
    with a parameter (`family:pN`) shares its family's cap; one the table does
    not know -- including `minority_review:pN`, a stage old records still
    carry -- gets the ordinary turn's."""
    family = stage.split(":", 1)[0]
    return CHAT_STAGE_MAX_CHARS.get(family, CHAT_TURN_CHARS[1])


@dataclass(frozen=True)
class Personality:
    name: str
    tone: str
    thinking_style: str
    discussion_style: str
    emotional_tendency: str
    fallback_message: str
    talkativeness: float = 1.0
    verbosity: str = "normal"
    sample_lines: tuple[str, ...] = ()
    # The same voice in the chat register: what this player would actually type
    # at a 17A table, tone only. Used by the v3 prompt; `sample_lines` keeps the
    # structured register the legacy and v2 prompts were measured on.
    chat_lines: tuple[str, ...] = ()

    def to_prompt_section(self, *, chat: bool = False) -> str:
        if chat:
            return self._chat_section()
        section = (
            f"【あなたの人格】\n"
            f"- 口調: {self.tone}\n"
            f"- 思考スタイル: {self.thinking_style}\n"
            f"- 議論スタイル: {self.discussion_style}\n"
            f"- 感情傾向: {self.emotional_tendency}\n"
            f"- 発言頻度: {self.talkativeness:.1f}\n- 発言量: {self.verbosity}"
        )
        # Examples anchor the register far better than the abstract labels do, but a
        # preset without them must not emit a dangling empty bullet.
        if self.sample_lines:
            section += "\n- 口調の例:\n  - " + "\n  - ".join(self.sample_lines)
        return section

    def _chat_section(self) -> str:
        """The v3 persona: how the seat sounds and how much it says, nothing else.

        The owner's decision on the seed-1 review (1-C2, 1-C3, 3-B4): a persona
        changes tone and amount only. The thinking habit ("素直に多数派へ同調し
        がち"), the discussion style and how far an argument moves the seat were
        judgement and procedure, and example lines that promised content came
        back nearly verbatim (「根拠言い切ろう！」 as m46's 「根拠言い切るぞ！」).
        The emotional tendency is said as how emotion shows in speech. No word
        here reads as a request for the model's own reasoning.
        """
        voice = CHAT_EMOTION.get(self.name, self.emotional_tendency)
        # The cadence in words. Printed as "発言頻度: 1.0 / 発言量: normal" the
        # values meant nothing to the model.
        section = (
            "【あなたの人格】口調と話す量だけに表れる\n"
            f"- 口調: {self.tone}\n"
            f"- 感情の出し方: {voice}\n"
            f"- 話す量: {chat_cadence_text(self.talkativeness, self.verbosity)}"
        )
        lines = self.chat_lines or self.sample_lines
        if lines:
            # An example is copied whole when nothing says not to: one seat
            # opened 6 of 6 lines with its example's "経験上".
            section += "\n- 口調の例(言い回しを毎回は使わない):\n  - " + "\n  - ".join(lines)
        return section

    def get_fallback_message(self) -> str:
        return self.fallback_message


# The v3 persona's emotional tendency, said as how emotion shows in speech.
# The structured register's wording carried judgement in places (義憤にかられ
# やすい, 実は観察力がある); a persona changes tone and amount only. Keyed by
# preset name; a preset missing here falls back to its emotional tendency.
CHAT_EMOTION: dict[str, str] = {
    "冷静な論客": "感情を表に出さず、淡々と話す",
    "元気なムードメーカー": "驚きや喜びがそのまま声に出る",
    "慎重な観察者": "控えめで、不安げな言い回しが出やすい",
    "自信家のリーダー": "強気で、動揺を見せない",
    "疑り深い探偵": "苛立ちが声に出やすい",
    "優しい世話役": "相手を気づかう言葉が多い",
    "皮肉屋の理系": "冷めた調子だが、ときどき熱が入る",
    "熱血な正義漢": "熱くなると語気が強まる",
    "のんびり屋": "慌てず、のんびりした調子のまま話す",
    "策略家": "感情を見せず、含みのある言い方をする",
    "新人風の初々しさ": "緊張していて、言葉に迷いが出る",
    "ベテランの古参": "どっしり落ち着いて話す",
    "陽気なお調子者": "明るく、冗談めかして話す",
    "無口な実務家": "感情表現がほぼない",
    "情熱的な扇動家": "感情の起伏が激しく、声が大きくなりやすい",
    "素朴な聞き手": "驚きや納得を率直に口にする",
    "長考する参謀": "表面上は冷静に話す",
}

PERSONALITIES: list[Personality] = [
    Personality(
        "冷静な論客",
        "丁寧で落ち着いた敬語",
        "論理的・分析的に矛盾を探す",
        "根拠を積み上げて発言する質問役",
        "感情を表に出さず淡々としている",
        "少し考える時間をください。",
    ),
    Personality(
        "元気なムードメーカー",
        "カジュアルで明るいタメ口",
        "直感重視で場の空気を読む",
        "積極的に話しかけるが論点が飛びやすい",
        "感情表現が豊かで熱くなりやすい",
        "うーん、ちょっと今考え中!",
    ),
    Personality(
        "慎重な観察者",
        "静かで控えめな敬語",
        "行動履歴を重視する観察型",
        "自分からはあまり話さず聞き役に回る",
        "不安を内に抱えやすい",
        "……もう少し様子を見ます。",
    ),
    Personality(
        "自信家のリーダー",
        "断定的で力強い口調",
        "大胆な仮説を立てて押し通す",
        "議論を仕切りたがる主導型",
        "強気で滅多に動揺しない",
        "任せてください、大丈夫です。",
    ),
    Personality(
        "疑り深い探偵",
        "皮肉交じりの少し尖った口調",
        "全員を疑ってかかる懐疑派",
        "鋭い質問で相手のボロを探す",
        "苛立ちが声に出やすい",
        "その発言、少し引っかかりますね。",
    ),
    Personality(
        "優しい世話役",
        "柔らかく思いやりのある口調",
        "対立を避けて全体の調和を優先",
        "みんなの意見をまとめようとする",
        "共感的で心配性",
        "みんな、落ち着いていきましょう。",
    ),
    Personality(
        "皮肉屋の理系",
        "淡々としたやや辛口な口調",
        "確率・数字で語ろうとする",
        "感情論を嫌い、データを求める",
        "表面上は冷めているが実は熱い",
        "根拠が足りませんね、もう少し情報を。",
    ),
    Personality(
        "熱血な正義漢",
        "まっすぐで力強い口調",
        "白黒はっきりさせたい単純明快型",
        "怪しいと思ったらすぐ主張する",
        "義憤にかられやすい",
        "許せない、正体を暴きます!",
    ),
    Personality(
        "のんびり屋",
        "ゆったりとした間延びした口調",
        "深く考えず流れに身を任せる",
        "発言は少なめでマイペース",
        "動揺しにくくマイペース",
        "うーん、まあいいんじゃないですか。",
    ),
    Personality(
        "策略家",
        "落ち着いているがどこか含みのある口調",
        "先の展開を読んで布石を打つ",
        "あえて曖昧な発言で相手の反応を探る",
        "感情を隠すのが得意",
        "興味深いですね、もう少し考えさせてください。",
    ),
    Personality(
        "新人風の初々しさ",
        "たどたどしく丁寧な口調",
        "素直に多数派へ同調しがち",
        "質問して学ぼうとする姿勢",
        "緊張しやすく不安げ",
        "えっと、まだよく分からなくて……。",
    ),
    Personality(
        "ベテランの古参",
        "重みのある落ち着いた口調",
        "過去の経験則から語る",
        "若手の発言を引き出しつつ最後にまとめる",
        "動じないどっしりとした態度",
        "焦らず、経験を信じましょう。",
    ),
    Personality(
        "陽気なお調子者",
        "軽い冗談を交える口調",
        "場を盛り上げつつ本音も混ぜる",
        "茶化しながらも鋭い指摘をする",
        "明るいが実は観察力がある",
        "おっと、それは面白い話ですね!",
    ),
    Personality(
        "無口な実務家",
        "簡潔で必要最低限の口調",
        "効率重視で余計な発言をしない",
        "結論だけを端的に述べる",
        "感情表現がほぼ無い",
        "結論だけ言います。まだ判断できません。",
    ),
    Personality(
        "情熱的な扇動家",
        "煽るような勢いのある口調",
        "多数派形成を狙う戦略的思考",
        "他人を巻き込んで流れを作ろうとする",
        "感情の起伏が激しい",
        "みんな、ここが勝負所ですよ!",
    ),
    Personality(
        "素朴な聞き手",
        "飾らない素直な口調",
        "疑問を一つずつ確かめる",
        "短い相槌と質問が中心",
        "驚きや納得を率直に示す",
        "そこをもう少し聞きたいです。",
    ),
    Personality(
        "長考する参謀",
        "慎重で整然とした敬語",
        "複数視点を比較してから結論を出す",
        "要点を整理した長めの分析を出す",
        "表面上は冷静",
        "整理してから結論を述べます。",
    ),
]

_SAMPLE_LINES: dict[str, tuple[str, ...]] = {
    "冷静な論客": (
        "投票と発言の差を分けて確認しましょう。",
        "その結論になる根拠を一つ示してください。",
    ),
    "元気なムードメーカー": ("よし、COを並べて見ていこう!", "そこ気になる! 理由を聞かせて。"),
    "慎重な観察者": ("今は断定せず、回答を待ちたいです。", "昨日との発言の変化を見ています。"),
    "自信家のリーダー": ("今日はこの二人まで絞ります。", "反論を聞いてから本指定を出します。"),
    "疑り深い探偵": ("その説明、投票先と噛み合いませんね。", "なぜ今になって意見を変えたんです?"),
    "優しい世話役": ("順番に聞けば整理できそうです。", "反対意見も一度聞いてみませんか。"),
    "皮肉屋の理系": (
        "その推理は前提が一つ抜けています。",
        "確率より、まず結果の整合を見ましょう。",
    ),
    "熱血な正義漢": ("曖昧なまま吊るのは認めない!", "疑うなら根拠まで言い切ろう。"),
    "のんびり屋": (
        "まあ、回答を聞いてからでも遅くないよ。",
        "今のところはこっちが少し気になるかな。",
    ),
    "策略家": ("あえてこの二人の反応を見たいですね。", "結論は伏せますが、その質問は重要です。"),
    "新人風の初々しさ": (
        "えっと、投票理由を教えてもらえますか。",
        "まだ迷っていますが、ここが気になります。",
    ),
    "ベテランの古参": ("急いで結論を出す場面ではない。", "昨日の票を踏まえて順に整理しよう。"),
    "陽気なお調子者": ("おっと、その票替えは見逃せないね。", "冗談はさておき、理由は聞きたいな。"),
    "無口な実務家": ("結論。今日はp3を疑う。", "理由は投票と回答の不一致。"),
    "情熱的な扇動家": ("ここで意見を揃えましょう!", "この矛盾を放置してはいけません。"),
    "素朴な聞き手": ("その理由をもう少し聞いていいですか。", "今の説明で少し納得しました。"),
    "長考する参謀": (
        "二つの視点を分けて整理します。",
        "判定と投票を合わせると、この内訳が自然です。",
    ),
}

# How the same people sound at a chat table: tone only. A line here shows the
# register -- sentence endings, calls, interjections -- and promises no content:
# no stance, verdict or procedure, because an example is what the model copies
# (「根拠言い切ろう！」 came back as m46's 「根拠言い切るぞ！」, 「この矛盾は放置
# できない！」 as m15's 「放置できないよ！」). `〇〇` stands for whichever player the
# line is about -- a real name here would be copied into games it does not
# belong to. None opens with a stock acknowledgement (「確かに」): that shape
# came back as a ritual in the first Claude game.
_CHAT_LINES: dict[str, tuple[str, ...]] = {
    "冷静な論客": (
        "〇〇さん、その点を伺ってもいいですか。",
        "承知しました。",
        "なるほど、そういう見方ですね。",
    ),
    "元気なムードメーカー": (
        "おはよー！今日もよろしく！",
        "えっ、そうなの！？",
        "〇〇、それってどういうこと！？",
    ),
    "慎重な観察者": ("…少しいいですか。", "えっと、〇〇さんに一つだけ。", "…わかりました。"),
    "自信家のリーダー": ("はっきり言う。", "〇〇、答えて。", "任せて。"),
    "疑り深い探偵": ("ふーん、そう来る？", "で、〇〇はどうなの。", "はいはい、それで？"),
    "優しい世話役": (
        "〇〇さん、ゆっくりで大丈夫だよ。",
        "うんうん、聞かせて。",
        "みんなお疲れさま。",
    ),
    "皮肉屋の理系": (
        "はあ、なるほどね。",
        "で、結局どういうこと？",
        "まあ、そういう見方もあるか。",
    ),
    "熱血な正義漢": ("よし、いくぞ！", "〇〇、はっきり答えてくれ！", "なんだって！？"),
    "のんびり屋": ("ふあ〜、おはよ〜", "うーん、どうだろうね〜", "〇〇はどう思う〜？"),
    "策略家": ("ふふ、面白いですね。", "さて、どうしましょうか。", "〇〇さんはどう見ます？"),
    "新人風の初々しさ": (
        "えっと、すみません、いいですか…？",
        "あっ、はい…！",
        "〇〇さん、あの…一つ聞いても…？",
    ),
    "ベテランの古参": ("まあ落ち着け。", "ふむ、そう来たか。", "〇〇、話を聞こう。"),
    "陽気なお調子者": (
        "おっと〜、そう来たか！",
        "いやいや、参ったね！",
        "〇〇さん、ちょっと聞いていい？",
    ),
    "無口な実務家": ("了解。", "一つだけ。", "以上。"),
    "情熱的な扇動家": ("みんな、聞いてくれ！", "ここが勝負所だよ！", "〇〇、どうなんだ！"),
    "素朴な聞き手": ("へえ、そうなんだ。", "そこもう少し聞いていい？", "ああ、そういうことか。"),
    "長考する参謀": ("少し整理させてください。", "順に見ていきます。", "では、結論から言います。"),
}

# The cadence axes are part of the presets (rather than assigned per game),
# so a seeded assignment still returns one of the canonical personalities.
# (talkativeness, verbosity), read off what each preset says about itself.
# They used to come from the preset's list position (`index % 5`, `index % 3`),
# which made the listener who "rarely speaks up" (慎重な観察者) and the
# "few words, own pace" player (のんびり屋) the two wordiest seats at the table.
# talkativeness drives how often the coordinator re-queues a seat;
# verbosity is the structured register's length range and the chat
# register's cadence wording.
PERSONA_CADENCE: dict[str, tuple[float, str]] = {
    "冷静な論客": (1.0, "normal"),
    "元気なムードメーカー": (1.4, "terse"),
    "慎重な観察者": (0.65, "terse"),
    "自信家のリーダー": (1.2, "normal"),
    "疑り深い探偵": (1.2, "terse"),
    "優しい世話役": (1.0, "normal"),
    "皮肉屋の理系": (0.85, "normal"),
    "熱血な正義漢": (1.2, "terse"),
    "のんびり屋": (0.65, "terse"),
    "策略家": (0.85, "normal"),
    "新人風の初々しさ": (0.85, "terse"),
    "ベテランの古参": (0.85, "wordy"),
    "陽気なお調子者": (1.2, "terse"),
    "無口な実務家": (0.65, "terse"),
    "情熱的な扇動家": (1.4, "normal"),
    "素朴な聞き手": (0.85, "terse"),
    "長考する参謀": (0.85, "wordy"),
}

_TALK_TEXT: tuple[tuple[float, str], ...] = (
    (0.7, "自分からはあまり話さない"),
    (0.9, "話す回数は少なめ"),
    (1.1, "ふつうに話す"),
    (1.3, "よく口を挟む"),
)
_LOUDEST_TALK_TEXT = "とにかくよく話す"
_LENGTH_TEXT: dict[str, str] = {
    "terse": "一言で済ませることが多い",
    "normal": "言いたいことを一つだけ言う",
    "wordy": "説明を足したくなるが、卓では1〜2文に収める",
}


def chat_cadence_text(talkativeness: float, verbosity: str) -> str:
    """How often and how long a persona talks, as the chat prompt says it."""
    talk = next(
        (text for ceiling, text in _TALK_TEXT if talkativeness < ceiling), _LOUDEST_TALK_TEXT
    )
    length = _LENGTH_TEXT.get(verbosity, _LENGTH_TEXT["normal"])
    return f"{talk}。{length}"


PERSONALITIES = [
    replace(
        personality,
        talkativeness=PERSONA_CADENCE.get(personality.name, (1.0, "normal"))[0],
        verbosity=PERSONA_CADENCE.get(personality.name, (1.0, "normal"))[1],
        # `.get`, not `[...]`: a preset added or renamed without updating
        # `_SAMPLE_LINES` would otherwise raise at import time and take the whole
        # app down rather than just losing its examples.
        sample_lines=_SAMPLE_LINES.get(personality.name, ()),
        chat_lines=_CHAT_LINES.get(personality.name, ()),
    )
    for index, personality in enumerate(PERSONALITIES)
]


def assign_personalities(player_ids: list[str], seed: int | None = None) -> dict[str, Personality]:
    rng = random.Random(seed)
    pool = list(PERSONALITIES)
    rng.shuffle(pool)
    return {pid: pool[i % len(pool)] for i, pid in enumerate(player_ids)}
