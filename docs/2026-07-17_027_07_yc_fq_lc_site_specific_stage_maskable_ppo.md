# 027_07 YC/FQ/LC 站点专属阶段型 MaskablePPO 记录

## 执行边界

- 三站点串行执行；未做联合训练、权重迁移或跨年验证。
- PPO 核心参数、9动作、I120/N300预算与 reward 结构未修改。
- YC使用官方六个可执行阶段7/30/45/60/80/100；FQ和LC因在DAP100前收获，仅使用真实可执行的7/30/45/60/80，未移动第六阶段。
- 原027_00 primary用于训练扩展硬门槛；导师最新至少一项严格领先视图只做报告，不参与reward或选模。

## 站点分支

_无数据_

## 预注册选中模型

_无数据_

## 失败记录

| site | year | error_type | error | traceback |
| --- | --- | --- | --- | --- |
| YC | 2014 | TypeError | Object of type bool_ is not JSON serializable | Traceback (most recent call last):
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 1108, in main
    result = run_site(spec, out_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 991, in run_site
    readiness = run_readiness(spec, site_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 734, in run_readiness
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
  File "/usr/lib/python3.10/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "/usr/lib/python3.10/json/encoder.py", line 201, in encode
    chunks = list(chunks)
  File "/usr/lib/python3.10/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/usr/lib/python3.10/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool_ is not JSON serializable
 |
| FQ | 2016 | TypeError | Object of type bool_ is not JSON serializable | Traceback (most recent call last):
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 1108, in main
    result = run_site(spec, out_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 991, in run_site
    readiness = run_readiness(spec, site_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 734, in run_readiness
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
  File "/usr/lib/python3.10/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "/usr/lib/python3.10/json/encoder.py", line 201, in encode
    chunks = list(chunks)
  File "/usr/lib/python3.10/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/usr/lib/python3.10/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool_ is not JSON serializable
 |
| LC | 2010 | TypeError | Object of type bool_ is not JSON serializable | Traceback (most recent call last):
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 1108, in main
    result = run_site(spec, out_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 991, in run_site
    readiness = run_readiness(spec, site_root)
  File "/workspace/src/run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07.py", line 734, in run_readiness
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
  File "/usr/lib/python3.10/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "/usr/lib/python3.10/json/encoder.py", line 201, in encode
    chunks = list(chunks)
  File "/usr/lib/python3.10/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.10/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/usr/lib/python3.10/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool_ is not JSON serializable
 |

## 结论边界

本记录只回答三个训练锚点在冻结阶段型MaskablePPO下能否产生跨seed初步信号。任何未达到2/3原primary的站点均按预注册规则停止，不据此调参；任何达到的站点也尚未完成同站跨年泛化。
