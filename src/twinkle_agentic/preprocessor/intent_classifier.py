# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

logger = get_logger(only_local_master=False)

# ── Intent categories ─────────────────────────────────────────────────────────
INTENT_TOOL_CALL = 'tool_call'
INTENT_CODE = 'code'
INTENT_MATH = 'math'
INTENT_USER_DISSATISFACTION = 'user_dissatisfaction'
INTENT_OTHER = 'other'

# ── Heuristic patterns ────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r'```[\s\S]{10,}?```')
_CODE_KEYWORD_RE = re.compile(
    r'\b(def |class |import |from |function |const |let |var |return |if \(|for \(|while \(|'
    r'#include|public class|private |protected |async |await |yield |throw |throws |catch |'
    r'switch |case |break |continue |void |struct |enum |interface |abstract |static |final |'
    r'namespace |package |module |export |lambda |func |fn |println|console\.log)\b|'
    # Symbolic call / arrow signatures occur even without the keywords above.
    r'(?:[a-zA-Z_]\w*\([^)\n]*\)\s*\{|=>\s*\{|->\s*[A-Za-z_]\w*)'
)

_MATH_LATEX_RE = re.compile(
    r'(\$\$.+?\$\$|\$[^$\n]+?\$|'
    r'\\frac|\\sum|\\int|\\lim|\\begin\{(equation|align|matrix)|'
    r'\\mathbb|\\partial|\\nabla|\\sqrt|\\overline|'
    r'\\boxed|\\text\{|\\mathrm|\\langle|\\rangle|\\cdot|'
    r'\\times|\\div|\\pm|\\leq|\\geq|\\neq|\\approx|\\equiv|'
    r'\\infty|\\pi|\\alpha|\\beta|\\gamma|\\theta|\\lambda|\\mu|\\sigma|\\prod|\\to|\\rightarrow|'
    r'\\\[.+?\\\]|'
    # R1-distill writes math in plain Unicode without $...$; catch operators, Greek, sub/super digits, fractions.
    r'[×÷±°∑∏∫√∂∇∞∈∋⊂⊃⊆⊇≤≥≠≈≡≅∝⇒⇔]|'
    r'[α-ωΔΘΛΞΠΣΦΨΩ]|'
    r'[⁰¹²³⁴-⁹₀-₉]|'
    r'[½⅓⅔¼¾⅛⅜⅝⅞]|'
    # Arithmetic equation pattern catches '30 ÷ 6 = 5' even when other markers are absent.
    r'\d+\s*[×÷\*/\+\-]\s*\d+\s*=\s*\d+|'
    # ≥4 comma-separated integers — number-sequence pattern.
    r'\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+|'
    # 'x = 5' / 'a = -3' style assignment.
    r'[a-zA-Z]\s*=\s*-?\d+|'
    # Chinese math vocabulary (strong indicators; ≥2 hits required so single occurrences in non-math text are safe).
    r'积分|微分|导数|求导|偏导|梯度|极限|矩阵|向量|行列式|特征值|特征向量|'
    r'多项式|因式分解|不等式|方程组?|二次方程|线性方程|求解|解方程|未知数|化简|约分|通分|因式|代入|应用题|算式|算术|计算题|一元(?:一次|二次|三次|方程|不等式|多项式)|二元(?:一次|二次|方程)?|'
    r'平方|立方|开方|根号|对数|指数函数|三角函数|正弦|余弦|正切|余切|反三角|'
    r'概率|期望值?|方差|标准差|分布|随机变量|均值|中位数|众数|百分比|比例|比率|'
    r'子集|并集|交集|空集|集合|映射|'
    r'乘以|除以|平方根|立方根|平方米|立方米|'
    r'系数|常数项|首项|项数|公差|公比|'
    r'切线|法线|渐近线|对称轴|双曲线|抛物线|椭圆|'
    # Geometry.
    r'三角形|四边形|多边形|长方形|正方形|圆形|圆锥|圆柱|球体|平行四边形|梯形|菱形|'
    r'半径|直径|周长|面积|体积|对角线|内角|外角|锐角|钝角|直角|平角|余角|补角|勾股|弧度|象限|坐标系|'
    # Sequences / number theory / elementary math.
    r'数列|数字序列|等差数列|等比数列|等差|等比|通项|递推公式|'
    r'奇数(?:位|项)?|偶数(?:位|项)?|质数|素数|合数|整数|小数|分数|有理数|无理数|实数|'
    r'因数|倍数|公因数|公倍数|最大公约数|最小公倍数|阶乘|排列组合|'
    r'余数|商(?=是|为|等)|被除数|除数|被乘数|乘数|'
    r'(?:加|减|乘|除)\d+|'
    r'第\d+(?:位|项)|'
    # English math vocabulary.
    r'\b(integral|differential|derivative|gradient|polynomial|equation|inequality|'
    r'matrix|vector|determinant|eigenvalue|eigenvector|coefficient|'
    r'logarithm|exponential|sqrt|theorem|lemma|proof|qed|axiom|corollary|'
    r'sine|cosine|tangent|cosecant|secant|cotangent|arcsin|arccos|arctan|'
    r'probability|variance|expectation|distribution|stddev|deviation|median|mean|mode|'
    r'subset|superset|union|intersection|multiply|divide|squared|cubed|factorial|'
    r'radius|diameter|circumference|perimeter|hypotenuse|congruent|parallel|perpendicular)\b|'
    r'\w_\{[^}]+\}|\w\^\{[^}]+\})',
    re.DOTALL,
)

_DISSATISFACTION_ZH_RE = re.compile(
    # Quality / correctness complaints.
    r'不[满好对行准确靠谱严]|不太[行好对准]|不正确|不准确|不对劲|不靠谱|不严谨|'
    # Severity intensifiers.
    r'太(差|慢|烂|傻|笨|垃圾|菜|弱|水|差劲|low)|这(么)?(差|烂|垃圾|傻|破|low)|'
    # Redo / retry.
    r'重[做来新答试]|重新(回答|做|来|算|想|考虑|生成)|再(答|来|做|算|想|试)一(次|遍|回|下)|你再答|'
    # Wrong / errors.
    r'错了?|错误|又错|搞错|弄错|出错|完全错|全错|大错|根本不(对|是)|压根不(对|是)|'
    # Off-topic / unhelpful.
    r'有问题|没用|没帮助|答非所问|文不对题|牛头不对|风马牛|跑题|偏题|偏离|跑偏|'
    # Stop talking nonsense.
    r'别瞎|别乱|别胡|你在说(什么|啥)|这是什么|这都什么|'
    r'离谱|搞什么|质量(太|很差)|胡(说|扯|言|乱|写|编|闹)|瞎(编|说|扯|写|想|猜|蒙|讲)|'
    # Random / illogical.
    r'莫名其妙|一塌糊涂|一派胡言|谬(论|误)|废话|屁话|没逻辑|没道理|说不通|不合逻辑|'
    # Negative emotion.
    r'不(满意|开心|高兴)|失望|让(我|人)失望|烦人|真烦|厌|气死|'
    # Misunderstanding / model failure.
    r'你(没|不)(懂|理解|明白|听懂)|理解错|抓不住重点|没get|没get到|'
    r'我说的不是|我问的不是|这不是我(说|问|想|要)|你听(错|不懂)|没听懂|'
    # Time / value waste.
    r'浪费时间|没意义|没价值|垃圾|废物|'
    # Generic anger.
    r'什么(玩意|东西|鬼)|你这是|你这答',
)
_DISSATISFACTION_EN_RE = re.compile(
    # Negative adjectives.
    r'\b(wrong|incorrect|useless|terrible|awful|horrible|bad|poor|lousy|sloppy|stupid|dumb|'
    r'idiotic|ridiculous|broken|misleading|infuriating|annoying|disappointing|disappointed|'
    r'unacceptable|unhelpful|inaccurate|imprecise|sub[- ]?par|low[- ]?quality)\b|'
    # "not X" complaints.
    r'\bnot (correct|right|good|helpful|useful|accurate|relevant|making sense|'
    r'what (i|I) (asked|wanted|meant|need|expected|requested))\b|'
    # Negation phrasings.
    r'(doesn\'?t|does not|didn\'?t|did not) (make sense|work|help|fit|match|address)|'
    r'makes? (no|zero|little) sense|'
    # Redo / retry.
    r'\b(redo|try again|do (it|this|that) again|start over|start again|do over|do better|'
    r'once more|again from scratch)\b|'
    # Insults / bullshit.
    r'\b(nonsense|garbage|trash|crap|bullshit|bs|baloney|hogwash|gibberish)\b|'
    r'(low|poor|bad|terrible) quality|waste of (time|effort|energy)|'
    # Misunderstanding.
    r'you (misunderstood|don\'?t understand|didn\'?t (get it|understand|listen)|missed (the|my) point)|'
    r'that\'?s (not what|wrong|incorrect|terrible|garbage|nonsense|useless)|'
    # Profanity.
    r'\b(WTF|wth|what the (heck|hell|fuck))\b|'
    # Off-target.
    r'\b(off[- ]topic|missed the mark|way off|completely off|totally wrong|nowhere near)\b|'
    r'not (even|really|quite) (close|right|correct)|'
    # Sarcasm / disbelief.
    r'come on|are you (serious|kidding|joking|sure)|'
    r'\bfrustrat\w+\b',
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg_text(msg: Dict[str, Any]) -> str:
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(
            p.get('text', '') for p in c
            if isinstance(p, dict) and p.get('type') == 'text'
        )
    return ''


def _pair_assistant(messages: List[Dict[str, Any]], idx: int, role: str) -> Optional[int]:
    """Resolve which assistant idx represents the round that owns a signal at (idx, role)."""
    if role == 'assistant':
        return idx
    if role == 'user':
        for j in range(idx + 1, len(messages)):
            m = messages[j]
            if isinstance(m, dict) and m.get('role') == 'assistant':
                return j
    return None


# ── Intent detectors (extensible pipeline) ────────────────────────────────────

class IntentDetector:
    """Base class. Each subclass sets ``intent`` and implements ``__call__``.

    ``__call__(messages)`` returns a list of assistant indices (key rounds) that
    match this intent within the given trajectory. An empty list means no match.
    Set ``definitive = True`` so the pipeline short-circuits on this detector
    (used for hard signals such as tool calls).
    """

    intent: str = ''
    definitive: bool = False

    def __call__(self, messages: List[Dict[str, Any]]) -> List[int]:
        raise NotImplementedError


class _RegexDetector(IntentDetector):
    """Common scaffolding: scan messages, run ``_match`` on each text, pair to assistant."""

    role_filter: Optional[str] = None

    def _match(self, text: str) -> bool:
        return False

    def __call__(self, messages):
        rounds = set()
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            if self.role_filter and role != self.role_filter:
                continue
            text = _msg_text(m)
            if not text or not self._match(text):
                continue
            asst_idx = _pair_assistant(messages, idx, role)
            if asst_idx is not None:
                rounds.add(asst_idx)
        return sorted(rounds)


class ToolCallDetector(IntentDetector):
    """Mark every assistant turn that carries a ``tool_calls`` payload."""

    intent = INTENT_TOOL_CALL
    definitive = True

    def __call__(self, messages):
        return [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get('role') == 'assistant' and m.get('tool_calls')
        ]


class CodeDetector(_RegexDetector):
    intent = INTENT_CODE
    threshold = 3

    def _match(self, text):
        blocks = _CODE_BLOCK_RE.findall(text)
        if blocks:
            return True
        return len(_CODE_KEYWORD_RE.findall(text)) >= self.threshold


class MathDetector(_RegexDetector):
    intent = INTENT_MATH
    # Threshold 4 (not 2): asst replies in chemistry/biology/materials describe formulas
    # like CH₂/H₂O whose subscript-digit chars match `_MATH_LATEX_RE`. Bumping to 4 keeps
    # genuine math (which has many more matches) while rejecting incidental sub/superscript
    # noise from non-math knowledge questions.
    threshold = 4

    def _match(self, text):
        return len(_MATH_LATEX_RE.findall(text)) >= self.threshold


class UserDissatisfactionDetector(_RegexDetector):
    intent = INTENT_USER_DISSATISFACTION
    role_filter = 'user'

    def _match(self, text):
        return bool(_DISSATISFACTION_ZH_RE.search(text) or _DISSATISFACTION_EN_RE.search(text))

    def __call__(self, messages):
        # Dissatisfaction is a reaction — require at least one prior assistant turn.
        seen_assistant = False
        rounds = set()
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            if role == 'assistant':
                seen_assistant = True
                continue
            if role != 'user' or not seen_assistant:
                continue
            text = _msg_text(m)
            if text and self._match(text):
                asst_idx = _pair_assistant(messages, idx, role)
                if asst_idx is not None:
                    rounds.add(asst_idx)
        return sorted(rounds)


# ── Preprocessor ──────────────────────────────────────────────────────────────

class IntentClassifier(Preprocessor):
    """Annotate each trajectory with its primary intent and key-round indices.

    Pure-heuristic, no LLM. Each intent is a pluggable :class:`IntentDetector`;
    pass ``detectors=[...]`` to extend or override.

    Annotates per row::

        row['intent']                  # primary intent string
        row['user_data']['key_rounds'] # list[int] of assistant indices
        row['user_data']['intents']    # dict[int, str] per-round intent
    """

    DEFAULT_DETECTORS: List[IntentDetector] = [
        ToolCallDetector(),
        CodeDetector(),
        MathDetector(),
        UserDissatisfactionDetector(),
    ]

    def __init__(
        self,
        detectors: Optional[List[IntentDetector]] = None,
        intent_field: str = 'intent',
    ) -> None:
        super().__init__()
        self._intent_field = intent_field
        self._detectors = list(detectors) if detectors is not None else list(self.DEFAULT_DETECTORS)

    def _detect(self, messages: List[Dict[str, Any]]) -> Dict[int, str]:
        """Run detector pipeline; later detectors never override earlier intent on the same round."""
        round_intents: Dict[int, str] = {}
        for det in self._detectors:
            rounds = det(messages)
            if not rounds:
                continue
            for idx in rounds:
                round_intents.setdefault(idx, det.intent)
            if det.definitive:
                break
        return round_intents

    def __call__(self, rows) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        out = []
        for row in rows:
            row = dict(row)
            messages = row.get('messages')
            round_intents = (
                self._detect(messages) if isinstance(messages, list) and messages else {}
            )

            if round_intents:
                primary = Counter(round_intents.values()).most_common(1)[0][0]
                user_data = dict(row.get('user_data') or {})
                user_data['key_rounds'] = sorted(round_intents)
                user_data['intents'] = dict(round_intents)
                row['user_data'] = user_data
            else:
                primary = INTENT_OTHER

            row[self._intent_field] = primary
            out.append(row)

        dist = Counter(r[self._intent_field] for r in out)
        logger.info(f'[IntentClassifier] distribution: {dict(dist)}')
        return out
