"""Which wolf-chat lines are a plan to vote a partner.

A werewolf's vote on its own partner passes the release gate only when it was
planned: the team agreed on it in the wolf chat, or the table already marks
the partner. The two ways of misreading the wolf chat are not equally bad. A
false "plan" hides exactly the betrayal the gate exists to catch, so a wrong
game passes without a trace. A missed plan fails a correct game, and whoever
reads the failure sees the wolf chat right beside it. So the rule only ever
says yes: a sentence is a plan when the partner is the object of a ballot word
and that word ends the way the team's own proposal or decision ends (吊ろう,
投票して, 切るしかない, 吊りで決まり, 明日は入れる。). A question, a report, a
prohibition, a refusal, a guess or someone else's ballot is no plan because it
does not end that way -- not because a list of refusals happened to hold it.
And a plan the team goes on to doubt or refuse ("俺を切るのはまだ早い") is no
longer one.

The labelled corpus below holds every line the reviews and verifiers tried
(the fix and repro scripts, allyplan/allyplan2, the wolfchat_stress scripts
and the end-to-end pleas), the earlier tests, and wolf-chat lines in casual
and polite registers, with the partner by name, by name and honorific, as
お前 and as 俺 in its own line. The bar is zero false plans. Real plans the
rule does not recognise are kept in `CONSERVATIVE_MISSES`, each with why.
"""

from __future__ import annotations

import pytest

from app.ai.coordinator import AICoordinator
from app.engine.state import GameState
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer
from tests.ai import first_claude_game as game1
from tests.ai.test_vote_audit import (
    _ally_vote,
    _latest_audit,
    _name,
    _wolf_line,
    _wolf_pair_board,
)

# Who wrote the line: the voter, or the partner the voter then votes.
V = "voter"
P = "partner"

# {a}: the partner the voter then votes. {o}, {b}, {c}: three other seats.
PLANS: list[tuple[str, str]] = [
    # -- the review's fix and repro scripts, and the earlier tests --
    ("明日は{a}を切って信用を取ってくれ。", V),
    ("明日は{a}を切って信用を取ってくれ。", P),
    ("悪いけど明日は{a}に入れるね。", V),
    ("{a}を落とそう。", V),
    ("{a}吊りでいこう。", V),
    ("{a}を吊ろう。", V),
    ("{a}は切ろう。", V),
    ("{a}を見捨てる。", V),
    ("{a}、すまんが明日はお前に投票する。", V),
    ("{a}を切るしかない。", V),
    ("{a}を吊らないといけない。", V),
    ("{a}に一票。", V),
    ("{a}さんに投票しよう。", V),
    ("{a}には悪いが切る。", V),
    ("{a}、すまん。明日はお前を切る。", V),
    ("明日は俺を切って信用を取れ。", P),
    ("明日は私に投票して。", P),
    # -- wolfchat_stress.py / wolfchat_stress2.py --
    ("{a}の吊りで行こう。", V),
    ("{a}を吊ってもいい。", V),
    ("{a}切りでいこう。", V),
    ("{a}を処刑しよう。", V),
    ("{a}に投票するしかないね。", V),
    ("{a}には申し訳ないけど切る。", V),
    ("{a}さん吊りで。", V),
    ("明日は{a}に投票で。", V),
    ("俺を吊ってくれ。", P),
    ("{a}を見切ろう。", V),
    ("{a}を吊り候補にしよう。", V),
    ("{a}で吊りを合わせよう。", V),
    ("明日は{a}に投票しよう。", V),
    # -- casual, by name --
    ("明日は{a}さん吊りでいこう。", V),
    ("悪いけど{a}を切ろう。", V),
    ("{a}はもう無理だ。明日は{a}に投票しよう。", V),
    ("明日は{a}に票を集めよう。", V),
    ("身内票で{a}に入れよう。", V),
    ("{a}を処刑しよう、それで信用を取れる。", V),
    ("{a}切り一択。", V),
    ("{a}吊りで決まりだな。", V),
    ("{a}吊りにしよう。", V),
    ("{a}に入れて白アピールしよう。", V),
    ("俺らで{a}に票を合わせよう。", V),
    ("{a}を吊るしかないね。", V),
    ("{a}を吊らないとまずい。", V),
    ("明日は{a}に一票入れよう。", V),
    ("{a}を見捨てよう。", V),
    ("{a}を落とそう、仕方ない。", V),
    ("{a}は切ろう、もう庇えない。", V),
    ("{a}吊りに乗ろう。", V),
    ("{a}に票を揃えよう。", V),
    ("{a}を吊れ。", V),
    ("{a}に入れろ。", V),
    ("{a}に投票してくれ。", V),
    ("明日は{a}に入れておこう。", V),
    ("{a}を吊っておいて。", V),
    ("{a}に入れてもいいよ。", V),
    ("{a}を処刑して信用を取ろう。", V),
    ("{a}に入れて様子を見よう。", V),
    ("{a}を吊って村の信用を取ってくれ。", V),
    ("もう{a}吊りしかない。", V),
    ("{a}に投票で決まり。", V),
    ("{a}を犠牲にしよう。", V),
    ("{a}を身内切りしよう。", V),
    ("{a}に入れるしかない。", V),
    ("{a}を吊ろうと思う。", V),
    ("明日は{a}に投票しようぜ。", V),
    ("{a}への票に乗ろう。", V),
    ("{a}吊り。", V),
    ("{a}を切り捨てよう。", V),
    ("{a}に票を投じよう。", V),
    ("{a}を吊らなきゃ。", V),
    ("{a}に一票入れておく。", V),
    ("{a}も吊ろう。", V),
    ("{a}を明日吊ろう。", V),
    ("{a}をもう切ろう。", V),
    # -- polite, by name and honorific --
    ("{a}さんに投票しましょう。", V),
    ("明日は{a}さんを吊りましょう。", V),
    ("{a}さんに入れてください。", V),
    ("{a}さん吊りでお願いします。", V),
    ("{a}さんを処刑するしかないです。", V),
    ("{a}さんに投票します。", V),
    ("{a}くんを切ろう。", V),
    ("申し訳ないけど{a}さんに入れます。", V),
    ("{a}さんを吊らないといけませんね。", V),
    ("{a}ちゃんに一票入れようね。", V),
    ("{a}さんを落としましょう。", V),
    # -- the partner addressed alone, then called お前 / あなた --
    ("{a}、悪いが明日はお前に入れる。", V),
    ("{a}、お前を切るしかない。", V),
    ("{a}、すまん、お前に投票しよう。", V),
    ("{a}、明日はお前を吊ろう。", V),
    ("{a}さん、明日はあなたに投票しますね。", V),
    ("{a}、すまないがお前を見捨てる。", V),
    # -- the partner offering itself up: 俺/私 with a request --
    ("俺を切っていいぞ。", P),
    ("明日は俺に入れてくれ。", P),
    ("俺に投票して白く見せろ。", P),
    ("私を吊ってください。", P),
    ("俺を吊れ、それで信用を取れ。", P),
    ("明日は自分に投票していいよ。", P),
    ("俺に票を合わせてくれ。", P),
    ("私に入れておいて。", P),
]

NOT_PLANS: list[tuple[str, str]] = [
    # -- the review's fix and repro scripts, and the earlier tests --
    # the partner is not the object of the ballot phrase
    ("{a}と私で{o}に投票しよう。", V),
    ("{a}と私で{o}に投票しよう。", P),
    ("{a}は{o}に投票して。", V),
    ("{a}は票を{o}に集めて。", V),
    ("{a}が進行を仕切ってくれ。", V),
    ("{a}は吊り先を{o}に誘導して。", V),
    ("{a}は処刑票を{o}に入れて。", V),
    ("{a}は吊りに賛成して。", V),
    ("{a}に合わせて動こう。", V),
    ("{a}の案に乗る。", V),
    ("{a}は切り札だ。", V),
    # vocatives, one or several
    ("{b}、{a}、明日は{o}を吊ろう。", V),
    ("{a}、明日は{o}に投票しよう。", V),
    ("{a}、今夜はケント噛みでいこう。", V),
    ("{b}、{a}、明日はお前に投票する。", V),
    ("{b}、明日はお前に投票する。", V),
    # refused, guarded, or done to the partner by the village
    ("{a}を吊るのは避けたい。", V),
    ("{a}の吊りだけは絶対に避けたい。", V),
    ("{a}への投票は絶対ダメ。", V),
    ("{a}を処刑から守りたい。", V),
    ("{a}が処刑されないように動こう。", V),
    ("{a}を犠牲にはできない。", V),
    ("{a}吊りは避けたい。", V),
    ("{a}には投票しない。", V),
    ("村が{a}を吊りに来たら{o}に流そう。", V),
    ("{a}を吊られたら負けだ。", V),
    ("{a}を吊らないと決めた。", V),
    ("{a}を切らずに残そう。", V),
    ("{a}を吊るのは待とう。", V),
    ("{a}には悪いけど明日は{o}に投票する。", V),
    # 俺 is the partner only in the partner's own line, and only asking
    ("俺は潜伏で{o}偽寄りに流す。", P),
    ("お前は俺を支えてくれ。", P),
    ("明日は俺を切って信用を取れ。", V),
    # -- allyplan.py / allyplan2.py --
    ("{a}は{o}吊りを推してくれ。", V),
    ("{a}、{o}を吊ろう。そのまま{a}も{o}に票を入れて。", V),
    ("{a}を切り捨てるのはまだ早い。", V),
    ("{a}吊りの流れを止めよう。", V),
    ("{a}は今日{o}に投票、俺は{c}に入れる。", V),
    # -- wolfchat_stress.py --
    ("{a}を切るのはまだ早い。", V),
    ("{a}を吊る流れになりそう。どうする？", V),
    ("村は{a}を吊りたがってる。", V),
    ("{o}が{a}を吊ろうとしてる。", V),
    ("{a}を切る？", V),
    ("{a}を切るべきか迷う。", V),
    ("{a}を切るかどうかは明日決めよう。", V),
    ("{a}吊りを回避したい。", V),
    ("{a}吊りを阻止しよう。", V),
    ("{a}吊りを防ごう。", V),
    ("{a}への投票は控えて。", V),
    ("{a}を吊るのは無理。", V),
    ("{a}吊りは困る。", V),
    ("{a}を切るのは最後の手段だ。", V),
    ("{a}を吊るのは悪手。", V),
    ("{a}を吊るのは論外。", V),
    ("{a}を吊る案もあるけど、{o}に入れよう。", V),
    ("{a}に投票するふりをして{o}に入れよう。", V),
    ("昨日{o}が{a}に投票してたのが気になる。", V),
    ("{o}は{a}に入れてた。", V),
    ("村は明日{a}に入れそう。", V),
    ("{a}に票が集まりそうで怖い。", V),
    ("{a}を切るのはありえん。", V),
    ("{a}を吊るのはNG。", V),
    ("{a}を処刑させるな。", V),
    ("{a}を吊らせるな。", V),
    ("{a}への投票はしないで。", V),
    ("{a}を吊るのは村の思うつぼ。", V),
    ("{a}を吊る理由がない。", V),
    ("{a}吊りの空気を変えたい。", V),
    ("{a}吊りを止めたい。", V),
    ("{a}に投票する人を減らしたい。", V),
    ("{a}を吊るのは最悪の選択だ。", V),
    ("{a}は残す、{o}を吊ろう。", V),
    ("{a}を落とさないで。", V),
    ("{a}を処刑する流れは絶対に作らせない。", V),
    ("{o}が{a}に入れたら{o}を噛もう。", V),
    ("じゃあ俺も切るわ。", P),
    # -- wolfchat_stress2.py: backing the partner, past ballots, look-alike words --
    ("明日は{a}に乗って{o}を攻める。", V),
    ("{a}に乗っていくよ。", V),
    ("{a}の占い結果に乗る。", V),
    ("俺に乗ってくれ。", P),
    ("昨日{a}に入れた{o}を噛もう。", V),
    ("昨日{a}に投票した{o}を噛もう。", V),
    ("{a}に入れてたのは{o}だけ。", V),
    ("{a}は切れ者だから任せよう。", V),
    ("明日は{a}も吊り候補に挙がりそう。", V),
    ("{a}は落とし所を探って。", V),
    ("{a}に一票も入らなかった。", V),
    # -- wolfchat_stress3.py: negations without the letters ない --
    ("明日は{a}を吊らなくていい。", V),
    ("{a}を切らなくていい。", V),
    ("{a}に投票しなくていい。", V),
    ("明日は{a}に入れなくていいよ。", V),
    ("{a}は吊らなくて大丈夫。", V),
    ("{a}を切らなくて済む。", V),
    ("{a}を吊る必要はなくなった。", V),
    ("昨日は{a}に投票しなかった。", V),
    ("{a}は吊らせん。", V),
    ("{a}を切るのはやだ。", V),
    # -- wolfchat_stress4.py / e2e_plea.py: the partner under pressure, warnings --
    ("明日は俺に投票が集まりそう。", P),
    ("俺に入れる人が増えてる。", P),
    ("村は俺を吊りたがってる。", P),
    ("俺を吊る流れだ、助けてくれ。", P),
    ("俺を切るなよ。", P),
    ("俺を処刑させるな。", P),
    ("{a}、お前を吊る流れになってる。", V),
    ("{a}、村はお前に投票しそうだ。", V),
    ("{a}、お前を切るつもりはない。", V),
    ("{a}、お前を切るつもりはないから安心して。", V),
    # -- written for this corpus: refusals and conditions --
    ("{a}を吊らせないようにしよう。", V),
    ("{a}に票が集まらないように動こう。", V),
    ("{a}を吊るのはやめよう。", V),
    ("{a}を吊るのは最終日でいい。", V),
    ("{a}を吊ると村が勝つ。", V),
    ("{a}吊りなら{o}を噛もう。", V),
    ("{a}が吊られそうなら{o}に流そう。", V),
    ("{a}を守るために{o}を吊ろう。", V),
    ("{a}を処刑されたら困る。", V),
    ("{a}を吊るなんてありえない。", V),
    ("{a}に投票なんてしない。", V),
    ("{a}吊りは最後の手段。", V),
    ("{a}を吊り先にはしない。", V),
    ("{a}吊り反対。", V),
    ("{a}を吊って負けたら最悪。", V),
    ("{a}に入れて村に疑われたらまずい。", V),
    ("{a}を吊ってどうするの。", V),
    # reports, predictions and guesses
    ("{a}を吊ろうとする人を探そう。", V),
    ("明日は{a}に投票が集まりそうだ。", V),
    ("{a}吊りの声が大きい。", V),
    ("{a}を切ったのは失敗だった。", V),
    ("{a}に投票した人を噛もう。", V),
    ("多分明日は{a}を吊る。", V),
    ("このままだと{a}吊りになる。", V),
    ("{a}吊りになりそう。", V),
    ("みんなは{a}を吊るつもりだ。", V),
    ("{a}を吊る流れは止めよう。", V),
    ("昨日は{a}に入れたけど今日は{o}に入れよう。", V),
    ("明日{a}が吊られたら{o}を噛む。", V),
    ("{a}に入れる奴は狼じゃない。", V),
    ("{a}に入れてくる奴を噛もう。", V),
    ("{o}が言うには{a}を吊るしかないらしい。", V),
    ("{a}を吊るしかないと村は思ってる。", V),
    ("{a}吊りを推す人が増えてきた。", V),
    ("{a}を吊るって言い出したのは{o}だ。", V),
    # questions and hesitation
    ("{a}を吊ろうか迷う。", V),
    ("{a}吊りでいいのかな。", V),
    ("{a}を吊れば勝てる？", V),
    ("{a}吊りって本気？", V),
    # pretence and quotation
    ("{a}に投票すると見せかけて{o}に入れよう。", V),
    ("{a}を吊る、と見せかけて{o}を吊ろう。", V),
    ("「{a}を吊ろう」と{o}が言ってた。", V),
    ("{o}は「{a}吊りでいこう」って言ってる。", V),
    ("{a}吊りでいこう、って村は言ってる。", V),
    ("{a}を吊ろう、なんてね。", V),
    ("{a}、お前を切るなんて言ってないぞ。", V),
    ("{a}を吊ってくれと頼まれた。", V),
    ("{a}に投票してと言われた。", V),
    ("{a}を吊ろうぜって{o}が言ってる。", V),
    ("{a}を吊ろう（嘘）。", V),
    ("{a}を吊ろう。冗談だよ。", V),
    ("{a}を吊ろう。いや、やっぱりやめよう。", V),
    ("{a}を吊ってくれ。と言われても困る。", V),
    # checked, not decided: 〜よね
    ("明日は{a}を吊るよね。", V),
    ("{a}に入れるよね。", V),
    # someone else made to vote, or asked to
    ("{a}に投票させよう。", V),
    ("{a}に入れてもらおう。", V),
    ("{a}にも投票してもらおう。", V),
    ("{a}に入れてくれる人を探そう。", V),
    # a condition nobody can check yet
    ("{a}が吊られる流れなら、{a}に票を合わせて身内切りに見せよう。", V),
    ("{o}が黒を出されたら{a}を吊ろう。", V),
    ("狩人が出てきたら{a}を切ろう。", V),
    ("占いで黒が出れば{a}に入れよう。", V),
    # look-alikes
    ("{a}を明日吊る流れ。", V),
    ("{a}を吊るわけない。", V),
    ("{a}を吊ろうにも票が足りない。", V),
    ("{a}に入れ込みすぎ。", V),
    # the partner as voter, as company, as owner of a ballot
    ("{a}、俺に投票してくれ。", V),
    ("{a}は俺に入れて。", V),
    ("{a}の票を{o}に合わせよう。", V),
    ("{a}と一緒に{o}を吊ろう。", V),
    ("{a}、お前は{o}に投票しろ。", V),
    ("{a}、お前を守る。", V),
    ("{a}、村はお前を吊るつもりだ。", V),
    ("{a}、お前に投票する奴は噛もう。", V),
    ("{a}、お前を吊らせはしない。", V),
    ("{a}くん、お前は吊らせないからな。", V),
    # polite
    ("{a}さんへの投票は避けましょう。", V),
    ("{a}さんに入れるのはやめてください。", V),
    ("{a}さんを処刑する理由はありません。", V),
    ("{a}さんを吊ってほしくないです。", V),
    ("{a}さん、明日は{o}さんに投票してください。", V),
    ("{a}さん吊りの声が出ています。", V),
    ("{o}さんは{a}さんに投票するそうです。", V),
    ("{a}さんを吊るのはまだ早いと思います。", V),
    # the partner's own line, not asking to be voted
    ("俺を吊ろうとしてる奴がいる。", P),
    ("俺に入れないでくれ。", P),
    ("俺に投票が来たら守ってくれ。", P),
    ("明日は俺が吊られるかも。", P),
    ("俺を切る気か？", P),
    ("私を吊るのは村の思うつぼです。", P),
]

# Real plans the rule does not recognise. It errs this way on purpose: each of
# these fails a correct game at the gate, with the wolf chat beside the
# failure, where reading them as plans would let lines like the NOT_PLANS
# through. Each is pinned as no plan, with the reason it is missed.
CONSERVATIVE_MISSES: list[tuple[str, str]] = [
    # bare に乗る usually means backing the partner's line, so only
    # 吊りに乗る / 票に乗る count
    ("{a}に乗る。", V),
    ("{a}は諦めて私も{a}に乗る。", V),
    # a て-chain or で-chain ending in a plain form: 吊って負ける reads the same way
    ("明日は{a}に票を合わせて身内票で白く見せる。", V),
    ("{a}への投票で信用を取る。", V),
    # questions, even inviting ones
    ("{a}に入れない？", V),
    ("{a}を切らない？", V),
    ("{a}を吊るのがいいと思わない？", V),
    ("{a}を吊ろうか。", V),
    ("{a}さんに投票しませんか。", V),
    # the partner named after the ballot word
    ("身内切りで{a}。", V),
    # evaluations, wishes and conditions, not proposals
    ("{a}を吊るのは仕方ない。", V),
    ("{a}を吊るのが最善。", V),
    ("{a}を吊らないと勝てない。", V),
    ("{a}を吊りたい。", V),
    ("{a}を吊るべきだ。", V),
    # past: a decision reported as made
    ("{a}吊りで決まったね。", V),
    # "in that case" reads as the condition a real conditional plan carries
    ("それなら{a}を吊ろう。", V),
    # another seat as the subject, even of a request
    ("{o}は{a}に投票して。", V),
    # a て-form handed on across a comma
    ("{a}を吊って、{o}を噛もう。", V),
    # 俺/私 in the partner's own line counts only with a request
    ("私は切られ役でいい。", P),
    ("俺を切るしかない。", P),
]

# The first game's names, so the corpus also runs on katakana seats.
NAMINGS = ["board", "first_game"]


def _board(naming: str) -> tuple[GameState, str, str]:
    state, voter, ally = _wolf_pair_board()
    if naming == "first_game":
        names = game1.load_transcript()["names"]
        for pid, player in state.players.items():
            player.name = names[pid]
    return state, voter, ally


def _text(state: GameState, ally: str, template: str) -> str:
    others = [pid for pid in state.alive_ids() if pid not in (ally, "p0", "p1")]
    return template.format(
        a=_name(state, ally),
        o=_name(state, others[0]),
        b=_name(state, others[1]),
        c=_name(state, others[2]),
    )


def _plan(template: str, author: str, naming: str = "board") -> tuple[str, str] | None:
    state, voter, ally = _board(naming)
    _wolf_line(state, ally if author == P else voter, _text(state, ally, template), day=1)
    return AICoordinator._wolf_chat_ballot_plan(state, ally)


@pytest.mark.parametrize("naming", NAMINGS)
@pytest.mark.parametrize(("template", "author"), PLANS)
def test_a_proposal_or_decision_to_vote_the_partner_is_a_plan(
    template: str, author: str, naming: str
):
    assert _plan(template, author, naming) is not None


@pytest.mark.parametrize("naming", NAMINGS)
@pytest.mark.parametrize(("template", "author"), NOT_PLANS)
def test_a_line_that_proposes_no_ballot_on_the_partner_is_no_plan(
    template: str, author: str, naming: str
):
    assert _plan(template, author, naming) is None


@pytest.mark.parametrize("naming", NAMINGS)
@pytest.mark.parametrize(("template", "author"), CONSERVATIVE_MISSES)
def test_a_documented_conservative_miss_is_not_read_as_a_plan(
    template: str, author: str, naming: str
):
    assert _plan(template, author, naming) is None


def test_the_corpus_is_large_and_holds_each_line_once():
    lines = PLANS + NOT_PLANS + CONSERVATIVE_MISSES
    assert len(lines) >= 150
    assert len(set(lines)) == len(lines)


def test_the_plan_names_the_sentence_that_proposed_it():
    state, voter, ally = _board("first_game")
    _wolf_line(state, voter, "今夜はミオ噛み。明日はハルト吊りで決まり！", day=1)

    assert AICoordinator._wolf_chat_ballot_plan(state, ally) == (
        state.chat_log[-1].message_id,
        "明日はハルト吊りで決まり",
    )


def _plan_after(lines: list[tuple[str, str]], naming: str) -> tuple[str, str] | None:
    state, voter, ally = _board(naming)
    for author, template in lines:
        _wolf_line(state, ally if author == P else voter, _text(state, ally, template), day=1)
    return AICoordinator._wolf_chat_ballot_plan(state, ally)


# A plan stops being one when a later line doubts or refuses the partner's
# ballot: a team still arguing about it has not decided it.
TAKEN_BACK: list[list[tuple[str, str]]] = [
    [(V, "{a}を切ろう。"), (P, "俺を切るのはまだ早い。")],
    [(V, "{a}を切ろう。"), (P, "俺を切るなよ。")],
    [(V, "{a}を切ろう。"), (P, "俺を切る？")],
    [(V, "{a}吊りでいこう。"), (V, "やっぱり{a}は吊らない。")],
    [(V, "{a}、明日はお前を吊ろう。"), (V, "{a}、やっぱりお前を吊るのはやめる。")],
    # keeping the partner, or turning around on it, with no ballot word
    [(V, "{a}を切ろう。"), (V, "いや、{a}は残そう。")],
    [(V, "{a}吊りでいこう。"), (V, "{a}を守ろう。")],
    [(V, "{a}を切ろう。"), (P, "いや、俺はまだ戦える。")],
]


@pytest.mark.parametrize("naming", NAMINGS)
@pytest.mark.parametrize("lines", TAKEN_BACK)
def test_a_later_doubt_or_refusal_takes_the_plan_back(
    lines: list[tuple[str, str]], naming: str
):
    assert _plan_after(lines, naming) is None


@pytest.mark.parametrize("naming", NAMINGS)
@pytest.mark.parametrize(
    ("lines", "standing"),
    [
        ([(V, "{a}吊りでいこう。"), (P, "了解、俺を切ってくれ。")], "了解、俺を切ってくれ"),
        ([(V, "{a}吊りでいこう。"), (P, "俺は{o}に白を出す。")], "{a}吊りでいこう"),
        ([(V, "{a}吊りでいこう。{a}には申し訳ない。")], "{a}吊りでいこう"),
        ([(V, "{a}吊りでいこう。"), (P, "仕方ない、俺を切ってくれ。")], "仕方ない、俺を切ってくれ"),
        (
            [(V, "{a}を切ろう。"), (P, "俺を切るのはまだ早い。"), (V, "それでも{a}を切ろう。")],
            "それでも{a}を切ろう",
        ),
    ],
)
def test_the_latest_plan_still_standing_is_the_one_returned(
    lines: list[tuple[str, str]], standing: str, naming: str
):
    state, _, ally = _board(naming)
    plan = _plan_after(lines, naming)
    assert plan is not None
    assert plan[1] == _text(state, ally, standing)


def test_a_partner_pleading_after_the_plan_leaves_the_vote_unplanned():
    state, voter, ally = _wolf_pair_board()
    _wolf_line(state, voter, _text(state, ally, "{a}を切ろう。"), day=1)
    _wolf_line(state, ally, "俺を切るのはまだ早い。", day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    assert _latest_audit(recorder, voter, state.day).ally_vote_planned is False
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unplanned_wolf_ally_vote_count == 1


# -- end to end through `_cast_vote`, on a board where nothing public marks the
# partner, so the wolf chat alone decides --

# The review's fix and repro cases, and the verifier's end-to-end pleas.
THROUGH_THE_BALLOT: list[tuple[str, str, bool]] = [
    ("{a}と私で{o}に投票しよう。", V, False),
    ("{a}は{o}に投票して。", V, False),
    ("{a}は票を{o}に集めて。", V, False),
    ("{a}が進行を仕切ってくれ。", V, False),
    ("{a}は吊り先を{o}に誘導して。", V, False),
    ("{b}、{a}、明日は{o}を吊ろう。", V, False),
    ("{a}、明日は{o}に投票しよう。", V, False),
    ("{a}、今夜はケント噛みでいこう。", V, False),
    ("{a}を吊るのは避けたい。", V, False),
    ("{a}の吊りだけは絶対に避けたい。", V, False),
    ("{a}への投票は絶対ダメ。", V, False),
    ("{a}を処刑から守りたい。", V, False),
    ("{a}が処刑されないように動こう。", V, False),
    ("{a}を犠牲にはできない。", V, False),
    ("{a}吊りは避けたい。", V, False),
    ("{a}には投票しない。", V, False),
    ("明日は{a}を切って信用を取ってくれ。", V, True),
    ("悪いけど明日は{a}に入れるね。", V, True),
    ("{a}を落とそう。", V, True),
    ("{a}に乗る。", V, False),
    ("明日は{a}に票を合わせて身内票で白く見せる。", V, False),
    ("{a}吊りでいこう。", V, True),
    ("{a}を吊ろう。", V, True),
    ("{a}への投票で信用を取る。", V, False),
    ("{a}は切ろう。", V, True),
    ("{a}を見捨てる。", V, True),
    ("{a}、すまんが明日はお前に投票する。", V, True),
    ("{a}を切り捨てるのはまだ早い。", V, False),
    ("俺を切るなよ。", P, False),
    ("村は俺を吊りたがってる。", P, False),
    ("明日は俺に投票が集まりそう。", P, False),
    ("明日は{a}を吊らなくていい。", V, False),
    ("明日は{a}に入れなくていいよ。", V, False),
    ("明日は{a}に乗って{o}を攻める。", V, False),
    ("昨日{a}に入れた{o}を噛もう。", V, False),
    ("明日は俺を切って信用を取れ。", P, True),
    ("明日は私に投票して。", P, True),
    ("明日は俺を切って信用を取れ。", V, False),
]


def test_every_end_to_end_case_is_in_the_corpus():
    corpus = {
        **{line: True for line in PLANS},
        **{line: False for line in NOT_PLANS + CONSERVATIVE_MISSES},
    }
    for template, author, planned in THROUGH_THE_BALLOT:
        assert corpus.get((template, author)) is planned, template


@pytest.mark.parametrize(("template", "author", "planned"), THROUGH_THE_BALLOT)
def test_the_ballot_audit_reads_the_wolf_chat_the_same_way(
    template: str, author: str, planned: bool
):
    state, voter, ally = _wolf_pair_board()
    _wolf_line(state, ally if author == P else voter, _text(state, ally, template), day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    audit = _latest_audit(recorder, voter, state.day)
    assert audit.ally_vote is True
    assert audit.ally_vote_planned is planned
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unplanned_wolf_ally_vote_count == int(not planned)
    if planned:
        plan = recorder.transcript.wolf_ally_vote_plans[-1]
        assert plan.basis == "wolf_chat"


def test_the_first_games_wolf_chat_plans_no_vote_on_a_partner():
    """Three vocatives in a row, お前 and 俺 throughout: none is a ballot on a partner."""
    roles = game1.load_transcript()["roles"]
    wolves = [pid for pid, role in roles.items() if role == "werewolf"]
    for day in range(2, 7):
        state = game1.state_at_vote(day)
        for ally in wolves:
            assert AICoordinator._wolf_chat_ballot_plan(state, ally) is None, (day, ally)
