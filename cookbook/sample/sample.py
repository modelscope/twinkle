"""使用 Qwen3.5-4B-Condenser LoRA 对三类原始数据进行压缩的示例。

三个场景：
  1. Python 代码（短）
  2. 长度约 5120 字符的中文新闻文本
  3. 含混杂字符的网页 HTML 代码

除代码外的所有自然语言均为中文。压缩 LoRA 默认指向 ModelScope 上的
``ms://twinkle-kit/Qwen3.5-4B-Condenser``，即与 ``cookbook/exp/grpo_condensed.py``
所用 condenser 一致；可通过环境变量 ``LORA_PATH`` 覆盖。

启动方式::

    SAMPLER_GPUS=1 python cookbook/sample/sample.py
    SAMPLER_GPUS=2 python cookbook/sample/sample.py    # 张量并行
"""

import os
from typing import Any, Dict, List

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3.5-4B')
LORA_PATH = os.environ.get('LORA_PATH', 'ms://twinkle-kit/Qwen3.5-4B-Condenser')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))


# ──────────────────────────────────────────────────────────────────────
# Condenser 提示词（与训练时严格对齐，保留英文原文以匹配 LoRA 训练分布）
# ──────────────────────────────────────────────────────────────────────
CONDENSER_SYSTEM = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Now begin.
"""

CONDENSER_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Target length\n'
    'Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars '
    '(~50% of the source). If core facts fit in far fewer chars, output fewer. '
    'Never exceed the ceiling.\n\n'
    '## Passage\n{text}')


# ──────────────────────────────────────────────────────────────────────
# 场景 1：Python 代码片段（Dijkstra 单源最短路）
# ──────────────────────────────────────────────────────────────────────
PY_QUERY = '这段代码的功能、方法名、出入参是什么？其他人如何调用？'
PY_PASSAGE = '''import heapq
from typing import Dict, List, Tuple


def dijkstra(graph: Dict[str, List[Tuple[str, float]]], src: str) -> Dict[str, float]:
    """Single-source shortest path on a non-negative weighted graph.

    Args:
        graph: adjacency list, ``graph[u] = [(v, w), ...]`` with ``w >= 0``.
        src:   source node id; must be a key of ``graph``.

    Returns:
        Mapping from node id to its shortest distance from ``src``;
        unreachable nodes get ``math.inf``.

    Time:  O((V + E) log V) via a binary heap.
    Space: O(V) for the distance map and the priority queue.
    """
    dist: Dict[str, float] = {node: float('inf') for node in graph}
    dist[src] = 0.0
    heap: List[Tuple[float, str]] = [(0.0, src)]
    visited: set = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if w < 0:
                raise ValueError(f'negative weight on edge {u}->{v}: {w}')
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
    return dist


if __name__ == '__main__':
    g = {
        'A': [('B', 1.0), ('C', 4.0)],
        'B': [('C', 2.0), ('D', 5.0)],
        'C': [('D', 1.0)],
        'D': [],
    }
    print(dijkstra(g, 'A'))
'''


# ──────────────────────────────────────────────────────────────────────
# 场景 2：长篇中文新闻（约 5120 字符）
# ──────────────────────────────────────────────────────────────────────
NEWS_QUERY = '本次峰会可以学习到什么，总结出什么经验？'
NEWS_PASSAGE = """2026年5月10日上午，为期三天的“全球低空经济创新峰会暨城市级低空示范走廊启用仪式”在合肥滨湖国际会展中心闭幕。会议由国家发展改革委、工业和信息化部、中国民用航空局共同主办，安徽省人民政府承办，吸引了来自三十六个国家和地区的一千二百余名代表，其中包括十七位省部级官员、四十二家飞行器整机企业代表、九十一家产业链上下游企业、二十三家科研院所及七家国际行业协会。会议公布了《低空经济创新发展指数（2026）》、《城市低空运行规则白皮书（试行版）》和《低空安全能力评估通用框架》等三份核心文件，明确将合肥、深圳、苏州、广州、成都、青岛六城列为首批“低空经济综合改革试验区”，并宣布合肥滨湖至庐阳总长四十六公里的“环城低空示范走廊”当日正式投入运行。

按照规划，环城低空示范走廊由九条主干航线和十六条支线组成，主干航线最低离地高度一百二十米、最高三百米，支线最低六十米、最高一百八十米；全网部署一百八十二个固定起降点和六十座可移动塔台，覆盖政务、医疗、应急、物流、低空旅游、城市巡检等十类典型业务场景。走廊采用“一张网、两套链、三层防”的运行架构，统一接入安徽省低空运行管理平台，平台部署三百台分布式边缘节点和两套异地灾备数据中心，单日峰值并发架次设计能力为六千架，支持十秒级动态空域调整与三十毫秒级冲突告警。运行首日上午即完成首班医疗血液配送、首班跨区低空通勤、首班高速公路应急救援与首班低空观光飞行等四项标志性任务。

国家发展改革委副主任周楠在主旨演讲中表示，低空经济作为我国正在加快培育的战略性新兴产业，2025年市场规模已突破六千八百亿元，年复合增长率连续三年保持在百分之三十二以上；按照《低空经济创新发展指数（2026）》预测，到2030年市场规模有望突破三万亿元，将带动直接就业岗位约二百二十万个、间接就业八百万个。她强调，下一阶段的政策重点将集中在三件事上：一是推动空域分类改革落地，将三百米以下空域使用审批权限下放至省级；二是建立全国统一的低空飞行身份认证体系，以“一码通飞”形式整合飞行器编号、运营资质、保险信息；三是加快建设“低空气象－通信－导航－监视”四张网，2027年前在三十座中心城市完成基础设施全覆盖。

中国民用航空局副局长邵岩晖介绍，新版《城市低空运行规则白皮书（试行版）》对运行主体提出了五项硬性要求：飞行器须取得型号合格证或试飞许可、运行人须建立安全管理体系并通过年度审核、机长须持有相应等级有效执照、第三方责任险保额单架不得低于人民币五百万元、城市核心区运行须接入城市低空数据共享平台。白皮书首次明确了无人机与有人机融合运行规则，规定融合空域内电子围栏精度优于五米、上传频率不低于每秒十次、应急避让响应时间不大于八百毫秒。试行版将在合肥、深圳、苏州先行实施六个月，2027年1月起在六个综合改革试验区全面推广。

本次峰会期间，共有四十八家整机与零部件企业进行了集中签约，签约总金额三百一十七亿元人民币。其中，亿航智能与合肥市政府就eVTOL航空枢纽建设达成战略合作，未来三年将在合肥落地两座垂直起降中心、一座飞行器维修工厂；峰飞航空V2000CG无人货运飞机宣布与京东物流、顺丰速运组建“低空干线物流联盟”，2026年内开通合肥-武汉、合肥-南京两条三百公里级日常货运航线；时的科技E20 eVTOL正式获得中国民用航空局型号合格审定（TC）受理通知，成为国内第二个进入TC审定阶段的国产载人eVTOL机型；中国电信、中国移动联合发布“低空通信定制网络”，提供基于5G-A的厘米级定位与十毫秒级时延切片服务，首批接入合肥、深圳、苏州三市示范走廊。

中国科学技术大学、北京航空航天大学、南京航空航天大学、中国航发湖南动力机械研究所等四家单位联合发布了五项关键技术成果。其中，中科大研制的“星臂II号”分布式电推进系统单机连续可靠工作时长突破六千小时，能量密度达到每千克四百二十瓦时；北航团队公布国内首套适用于城市楼宇间复杂气流环境的“激流-3”自主感知与避障算法，已在合肥CBD连续完成八千架次实飞验证，避障成功率达百分之九十九点九七；南航发布的“穹顶”机载多源融合定位单元在GPS拒止环境下可实现亚米级定位，精度优于现有民用产品三倍。

国际合作方面，中欧双方在峰会上签署《低空运行互认合作备忘录》，约定2027年起对各自认证的两吨级以下载人电动飞行器互相承认型号合格证基础部分，争议技术指标通过联合评审解决。中国与阿联酋、新加坡、巴西、德国、日本五国民航主管部门签署双边谅解备忘录，覆盖低空气象数据互通、跨境物流走廊试点、飞行员资质互认三个方向。世界经济论坛代表在致辞中评价，合肥示范走廊是“迄今为止全球规模最大、运营规则最完整的城市级低空融合试验场”。

为保障示范走廊安全运行，安徽省专门组建了“低空安全联合运行中心”，由民航华东空管局、安徽省公安厅、应急管理厅、气象局以及合肥市政府五方常态派员，实行7×24小时值守。运行中心配备六十四套全向相控阵雷达、九十二套低空ADS-B接收机和一百二十组光电跟踪设备，可对覆盖空域内大于0.05平方米的目标进行毫秒级追踪；同时部署了五十架次自动巡查无人机和两架有人机巡查直升机，对低慢小目标实行混合编队拦截。运行首日，中心累计处置告警事件二十三起，其中误闯入九起、设备失联六起、超高飞行四起、外部气象突变三起、未授权改航一起，全部在三分钟内完成处置。

针对普通市民关心的应用场景，主办方在滨湖国际会展中心外侧搭建了占地约一万二千平方米的“低空生活体验区”。市民可通过现场或“合肥低空”小程序预约低空观光（合肥环城线，单程二十分钟，票价人民币二百九十八元）、低空通勤（滨湖至合肥南站，单程八分钟，票价九十八元）、低空配送（三公里内三十分钟达，订单费十二元）和低空应急医疗演示等四项体验。仪式当日预约平台一上线即满负荷运转，截至当天下午五点，累计提交订单超过一万一千笔，其中观光类占百分之六十四、通勤类占百分之二十二、配送类占百分之十三。

投融资方面，峰会同期举行的“低空产业投资人之夜”披露：2025年我国低空领域股权融资总额突破七百八十亿元，同比增长百分之七十二，融资轮次主要集中在A轮至C轮，平均单笔金额一点二亿元；其中飞行器整机、电池电机电控、空管软件三类标的占比分别为百分之四十一、百分之二十三、百分之十八。安徽省产业投资集团联合中国国新基金、中信产业基金、深创投、合肥兴泰金融等十家机构发起设立总规模二百亿元的“低空经济母基金”，首期规模六十亿元，重点投向飞行器适航取证、低空通信导航、城市运营平台与高端材料四个方向，单项目投资上限三亿元，预计三年内完成对外投放。

人才培养方面，国家民航局、教育部、人力资源和社会保障部联合发布《低空飞行人员培养行动计划（2026-2030）》，明确到2030年累计培养低空领域专业人才四十万人，其中eVTOL机长六万人、地面运行控制员八万人、无人机系统工程师十二万人、低空气象与运行支持人员六万人、产业链高端研发人员八万人。中国民航大学、中国民用航空飞行学院、合肥工业大学、深圳职业技术大学等十六所院校将于2026年秋季学期同步开设“低空运行与管理”本科专业和“无人飞行器系统工程”研究生方向，前两年招生总规模约六千八百人，并实行校企双导师制。

法治保障方面，《中华人民共和国低空空域使用管理条例（草案）》已于4月底完成第二轮社会公开征求意见，预计2026年下半年提交全国人大常委会审议。条例（草案）首次以法律形式确立“分类分级、责任清晰、动态管理”的空域使用原则，明确300米以下非管制空域备案准入、300米至1000米管制空域许可准入；规定运行人对所造成的人身、财产损害承担无过错责任，强制责任险最低保额按机型分为单架人民币三百万、五百万、一千万三档；对违规飞行的行政处罚上限从原《民用航空法》的人民币十万元提高至人民币一百万元，构成犯罪的依法追究刑事责任。

民众体验环节中，本报记者亲自试乘了由亿航EH216-S执飞的合肥环城观光航线。从滨湖国际会展中心垂直起降平台起飞，飞行器在二十秒内攀升至一百八十米高度，随后沿环城西线向北巡航，途经合肥南站、政务区、合肥植物园等地标，全程巡航速度九十公里每小时，最高速度一百一十公里每小时，舱内噪音实测六十六分贝、相当于普通会议室水平；地面起降阶段振动幅度小于零点二G，乘坐感受平稳。值得一提的是，飞行器全程由地面无人值守，机舱内仅有四枚乘客座位与一台显示飞行参数和航线进度的十英寸触控屏，乘客可一键切换中文、英文、日文三种语音解说。

技术展望部分，多位专家在分论坛中表达共识：未来五年制约低空经济规模化的关键不是飞行器性能，而是“运行密度天花板”——即在城市核心区如何把单位空域、单位时间内的安全飞行架次密度从当前的十架次每平方公里每小时提升到二百架次。中国工程院院士王建宇指出，要突破这一瓶颈必须解决三个核心问题：一是低慢小目标的全天候、全气象、全城域感知；二是冲突探测与解脱算法在高密度场景下的实时性，目标响应时间需压缩到二十毫秒以内；三是空地一体化通信网络的可用性，必须达到5个9（99.999%）的可靠度。预计这些核心技术将在“十四五”末取得阶段性突破，并在“十五五”实现产业化推广。

区域协同与产业布局方面，本次峰会同期发布了《中国低空经济产业布局白皮书（2026）》，首次以全国六十三个重点城市为样本，对产业链上下游进行了详尽画像。白皮书揭示，现阶段我国低空产业已初步形成“三极多点”的空间格局：长三角以合肥、南京、苏州、上海、杭州五市为核心，重点发展eVTOL整机、高端重载无人机与运营平台，产业营收占全国百分之三十五；珠三角以深圳、广州为核心，重点发展消费级与商业级无人机、低空物流，产业营收占百分之二十九；成渝地区以成都、重庆为核心，重点发展航空发动机、错复材料与航电系统，占百分之十三；其余千亿级“多点”包括青岛、沈阳、西安、武汉、长沙五个区域中心。白皮书同时提示，中西部低空产业发展仍存在不平衡问题，需重点加强边境州、边境县及山区、草原地区的低空基础设施补齐，预计这些区域有3200多个乡镇需新增低空起降点。为此，发改委将联合农业农村部、国务院应急管理部启动专项补助，中央财政三年内安排专项资金一百五十亿元。

闭幕仪式上，安徽省政府宣布将在2026年内追加投资八十亿元用于二期走廊建设，二期工程将向北延伸至淮南、向西延伸至六安，总长度从四十六公里扩展到一百九十六公里，预计2027年底前贯通。下一步合肥还将牵头编制《长三角低空一体化运行总体方案》，推动沪苏浙皖四省市在2028年前实现“一码通飞、一卡结算、一平台调度”的跨省域低空运行体系。中国民用航空局表示，将在2026年第三季度发布《国家低空经济发展中长期规划（2026-2035）》，明确未来十年的总体目标、重点任务、保障措施与考核机制。本次峰会的全部技术文件、签约项目清单和示范走廊运行实时数据将在“中国低空经济服务网”同步公开。据主办方最后公布的统计，为期三天的峰会共举办主论坛一场、高端对话三场、专题分论坛二十三场、企业发布会十六场，现场展示飞行器与装备总计一百八十八架（套），其中eVTOL三十二架、重载无人货运机二十七架、高端作业型无人机六十五架、空管与低空助航装备五十四套；累计进场观众超过二十三万人次，现场达成预订订单十二万五千余笔，为合肥市仪式期间酒店入住率、餐饮营业额分别带来同比百分之四十二与百分之三十六的增长。与会代表普遍评价，本届峰会首次实现了“会议、试点、产业、民生”四者同场并进，将原本存在于不同部门、不同会议、不同后续跨年的不同工作压缩为一个集中阶段，明显提高了这轮低空经济发展的政策准备度与社会可见度。据估计，在后续三个月内，首批六个“低空经济综合改革试验区”均将起步运行走廊与试点业务，预计2027年上半年可被看到首轮可复制、可推广的运行经验与产业样本。
"""


# ──────────────────────────────────────────────────────────────────────
# 场景 3：含混杂字符的网页 HTML（电商商品详情页）
# ──────────────────────────────────────────────────────────────────────
HTML_QUERY = '这段html代码的结构如何？如何使用js如何对接？'
HTML_PASSAGE = """<!DOCTYPE html>
<html lang="zh-CN" data-spm="product-detail">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>云端Air Pro 13 笔记本电脑（2026款）| TechMart 数码旗舰店</title>
  <meta name="description" content="云端Air Pro 13，搭载自研M3-Pro芯片，14核CPU+18核GPU，售价¥9,999，限时8折，赠AppleCare+">
  <meta property="og:price:amount" content="9999.00">
  <meta property="og:price:currency" content="CNY">
  <link rel="canonical" href="https://shop.techmart.cn/p/yda-pro-13-2026">
  <script>window.__INITIAL_STATE__ = {"sku":"YDA-PRO-13-2026","stock":237,"region":"CN-AH"};</script>
  <style>
    .price del{color:#999;text-decoration:line-through;margin-right:8px;}
    .price strong{color:#e60012;font-size:32px;font-weight:700;}
    .badge{background:#ff4d4f;color:#fff;padding:2px 6px;border-radius:4px;font-size:12px;}
  </style>
</head>
<body>
  <header class="nav"><a href="/">首页</a> &gt; <a href="/c/laptop">笔记本电脑</a> &gt; 云端Air Pro 13</header>
  <main>
    <h1>云端Air Pro 13&nbsp;<span class="badge">2026款 · 全网首发</span></h1>
    <p class="brand">品牌：云端 (Yunduan)&nbsp;|&nbsp;型号：YDA-PRO-13-2026&nbsp;|&nbsp;颜色：星空银 / 深空灰 / 沙漠金</p>
    <div class="price">
      <del>¥12,499.00</del>
      <strong>¥9,999.00</strong>
      <span>立省 ¥2,500，限时48小时</span>
    </div>
    <ul class="spec">
      <li>处理器：自研 M3-Pro，14核CPU @ 3.6GHz / 18核GPU / 16核NPU（35 TOPS）</li>
      <li>内存：18GB LPDDR5X-7500（统一内存架构）</li>
      <li>存储：512GB / 1TB / 2TB NVMe SSD（最高读取 7,400 MB/s）</li>
      <li>屏幕：13.6&Prime; Liquid Retina，2,560×1,664，600 nits 峰值，DCI-P3 100%</li>
      <li>电池：72Wh，续航最长 18h（本地视频播放）</li>
      <li>重量：1.24 kg&nbsp;|&nbsp;厚度：11.3 mm</li>
      <li>接口：2× Thunderbolt 5、1× HDMI 2.1、1× MagSafe 3、1× 3.5mm 耳机</li>
      <li>无线：Wi-Fi 7（802.11be）、蓝牙 5.4、UWB</li>
    </ul>
    <p>赠品（前 100 名下单）：原装 65W GaN 电源、Type-C&rarr;HDMI 2.1 转换线、防泼溅键盘膜、AppleCare+ 1 年延保。</p>
    <div class="promo">!! 限时优惠：叠加云端校园券 ¥500 + 以旧换新最高补贴 ¥1,200 !!</div>
    <section class="qa">
      <h2>常见问题</h2>
      <p>Q: 是否支持 Windows 11 ARM 双系统？&nbsp;A: 不支持，但可通过 Parallels Desktop 19 虚拟运行。</p>
      <p>Q: 发货时效？&nbsp;A: 现货 24 小时内发出，安徽/江苏/浙江/上海次日达。</p>
    </section>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product","name":"云端Air Pro 13",
     "sku":"YDA-PRO-13-2026","brand":{"@type":"Brand","name":"Yunduan"},
     "offers":{"@type":"Offer","price":9999.00,"priceCurrency":"CNY",
               "availability":"https://schema.org/InStock"}}
    </script>
  </main>
  <footer>&copy; 2026 TechMart Inc. 沪ICP备2021xxxx号 &middot; 京公网安备 31010102xxxxxx号</footer>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# 组装 prompts
# ──────────────────────────────────────────────────────────────────────
def build_prompts() -> List[Dict[str, Any]]:
    """构造三个场景的 Trajectory dict 列表。"""
    cases = [
        ('Python 代码', PY_QUERY, PY_PASSAGE),
        ('中文长篇新闻', NEWS_QUERY, NEWS_PASSAGE),
        ('网页 HTML', HTML_QUERY, HTML_PASSAGE),
    ]
    prompts: List[Dict[str, Any]] = []
    for tag, query, passage in cases:
        # 50% 硬上限，与训练时一致
        budget = max(1, len(passage) // 2)
        user_msg = CONDENSER_USER.format(query=query, budget=budget, text=passage)
        prompts.append({
            'tag': tag,
            'src_len': len(passage),
            'budget': budget,
            'messages': [
                {'role': 'system', 'content': CONDENSER_SYSTEM},
                {'role': 'user', 'content': user_msg},
            ],
        })
    return prompts


def main():
    # 1. 初始化 Twinkle + Ray
    device_groups = [
        DeviceGroup(name='sampler',
                    ranks=list(range(SAMPLER_GPUS)),
                    device_type='GPU',
                    gpus_per_worker=SAMPLER_GPUS),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=SAMPLER_GPUS, groups=device_groups)

    # 2. 构造 vLLMSampler，max_model_len 需容纳 5120 字符级原文 + 系统提示 + 输出
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': 16384,
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': 32,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    logger.info(get_device_placement())

    # 3. 采样参数：压缩任务用偏低温度，避免幻觉
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.4,
        top_p=0.9,
        num_samples=1,
    )

    # 4. 推理
    prompts = build_prompts()
    logger.info(f'共 {len(prompts)} 个压缩场景，模型 {MODEL_ID}，LoRA {LORA_PATH} ...')

    responses = sampler.sample(
        [{'messages': p['messages']} for p in prompts],
        sampling_params,
        adapter_path=LORA_PATH,
    )

    # 5. 输出结果
    for i, response in enumerate(responses):
        meta = prompts[i]
        for seq in response.sequences:
            text = seq.decoded
            logger.info(
                f'\n{"=" * 60}\n'
                f'场景 {i + 1}：{meta["tag"]}（原文 {meta["src_len"]} 字符，硬上限 {meta["budget"]} 字符）\n'
                f'{"-" * 60}\n'
                f'压缩结果（{len(text)} 字符）：\n{text}\n')

    logger.info('全部场景压缩完成。')


if __name__ == '__main__':
    main()
