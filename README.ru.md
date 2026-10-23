<div align="center">

```
 ┌─────────────────────────────────────────────┐
 │  BEGIN:VCARD                                │
 │  UID:hello                                  │
 │  FN:vcard-cpu                               │
 │  X-OP:MUL                                   │
 │  RELATED;TYPE=input:yourGPU,yourCPU         │
 │  END:VCARD                                  │
 └─────────────────────────────────────────────┘
```

# vcard-cpu

**vCard 4.0 как исполняемый контейнер вычислительных ядер.**
Да, *тот самый* vCard — формат контактных файлов. Ваши SIMD-ядра заслуживают
лучшего, чем самодельный YAML.

[![CI](https://github.com/example/vcard-cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/example/vcard-cpu/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green)]()

*English README: [README.md](README.md)*

</div>

---

## Коротко

`.vcf`-файл перестаёт быть контактной карточкой и становится **программой**.
Карточки с `X-OP` — это SIMD-узлы операций, связанные в DAG через свойства
`RELATED`. JIT транслирует весь граф в **один слитый AVX2-цикл**, компилирует
его gcc и вызывает поверх NumPy-буферов.

```console
$ vcard-cpu show kernels/graph.vcf
node-mult: op=MUL in=['in_x', 'in_y']
node-sub: op=SUB in=['node-mult', 'in_z']
node-abs: op=ABS in=['node-sub']
node-clamp: op=MIN in=['node-abs', 'in_limit']
inputs=['in_x', 'in_y', 'in_z', 'in_limit'] outputs=['node-clamp'] params=[]
--- generated C ---
#include <immintrin.h>
...
void execute_kernel(const float* in__in_x, ..., float* out__node__clamp, int n) {
    for (; i <= n - 8; i += 8) {          // 8 float за инструкцию
        __m256 t0 = _mm256_mul_ps(v_in__in_x, v_in__in_y);
        __m256 t1 = _mm256_sub_ps(t0, v_in__in_z);
        __m256 t2 = _mm256_and_ps(t1, sign_mask);
        __m256 t3 = _mm256_min_ps(t2, v_in__in_limit);
        _mm256_storeu_ps(&out__node__clamp[i], t3);
    }
    ...
}
```

Четыре vCard → одно безветвительное, регистр-в-регистр, полностью
векторизованное ядро. Без промежуточных буферов. Без Python в горячем цикле.

## Почему vCard?

Половина шутки, половина всерьёз:

1. **Модель графа уже есть.** `RELATED;TYPE=input:` — типизированное ребро,
   `UID` — идентификатор узла, `FN` — документация. RFC 6350 сделал скучную
   работу за нас.
2. **Параметры уже есть.** `X-PARAMS:high_th:float` становится настоящим
   аргументом C-функции — «номер телефона» вашего ядра это порог Шмитта.
3. **Человекочитаемые контейнеры лучше непрозрачных бинарников** для
   распространения индустриальных рецептов. Калибровочная кривая Pt100
   заслуживает версионирования в формате, который инженер осилит в поезде.
4. **Categories как метаданные развёртывания**:
   `CATEGORIES:Industrial,Sensors,SIMD` — это бесплатный индекс каталога
   ядер.

Серьёзное ядро под шуткой: **vCard — демонстрационный субстрат для схемы
«декларативный SIMD DSL → слитый C»**, того же трюка, что играют Mojo,
Halide и TVM — только в формате настолько нелепом, что конвейер виден
насквозь.

## Три способа сказать «вычисли»

### 1. Граф из op-карточек (авто-слияние)

```vcf
BEGIN:VCARD
UID:node-mult
X-OP:MUL
RELATED;TYPE=input:in_x,in_y
END:VCARD
...
```

Доступные операции: `ADD SUB MUL DIV MAX MIN ABS NEG SQRT RSQRT FMADD FMSUB RELU`.
Компилятор топологически сортирует DAG, порождает один слитый AVX2-цикл со
скалярным хвостом, компилирует и отдаёт Python-вызываемый объект.

### 2. Сырое тело ядра (принесите свой AVX2)

```vcf
BEGIN:VCARD
UID:isa-pt100-adc-calibrate
FN:vMODBUS_PT100_CONVERT
CATEGORIES:Industrial,Sensors,SIMD
RELATED;TYPE=input:adc
RELATED;TYPE=output:out
X-KERNEL-BODY:
// T = (ADC * 0.005) * (1.0 + ADC * 0.000001) - 50.0
__m256 v_scale = _mm256_set1_ps(0.005f);
...
END:VCARD
```

Всё между `X-KERNEL-BODY:` и `END:VCARD` — дословный C, встраиваемый в
сгенерированную функцию с корректной сигнатурой.

### 3. Python-выражение → AVX2

```console
$ vcard-cpu expr "a * b + c" -i a -i b -i c --n 500000
n=500000 elapsed=0.68 ms (~5.92 GOPS naive)
```

`vcard-cpu expr` разбирает выражение модулем `ast`, валидирует белым
списком, выносит числовые константы из горячего цикла в `__m256` и
опускает выражение через тот же реестр операций, что и компилятор графа.
Свёртка констант по всему выражению — бесплатно.

## Бенчмарки (честные цифры)

| Ядро                            | n         | Время    | Примечание                       |
|---------------------------------|-----------|----------|----------------------------------|
| Полином Pt100 (FMADD×2, FMSUB)  | 2 000 000 | 1.59 мс  | макс. ошибка против numpy: 6·10⁻⁵ °C |
| Слитый конвейер relu·rsqrt      | 2 млн     | ~1.9 мс  | `rsqrt` — приближённый AVX2-вариант |
| Выражение `a*b+c` через JIT     | 500 000   | 0.68 мс  | включая оверхед ctypes            |

Цифры с ноутбучного CPU (`cc -O3 -mavx2 -mfma`). Дело не в скорости — дело
в том, что **vCard стал скомпилированным SIMD-ядром быстрее чем за 2 мс**.

## Установка

```bash
git clone https://github.com/example/vcard-cpu
cd vcard-cpu
pip install -e .
vcard-cpu --help
```

Требования: Linux/macOS x86_64 с AVX2/FMA, gcc/clang в рантайме (JIT
компилирует через `cc -O3 -mavx2 -mfma`), Python 3.9+, numpy.
Собранные ядра кэшируются в `~/.cache/vcard_cpu/` по хэшу исходника.

## Справка по CLI

```
vcard-cpu show  <file.vcf>              карточки + план генерируемого C (без сборки)
vcard-cpu run   <file.vcf> [--n N]      компиляция + запуск на случайных входах
                                        [--param name=value] для X-PARAMS
vcard-cpu expr  "a*b+c" -i a -i b -i c  разовое ядро-выражение
                [--n N] [--emit-c]
```

## Python API

```python
import numpy as np
from vcard_cpu import VcfJit

jit = VcfJit(path="pt100.vcf")
print(jit.meta_inputs)    # ['adc']
out = jit.run({"adc": adc_array})["out"]
```

`VcfJit.run` проверяет наличие и длины входов, выделяет выходы, передаёт
скалярные `params` и возвращает обычные NumPy-массивы.

## Структура репозитория

```
vcard-cpu/
├── src/vcard_cpu/
│   ├── vcard.py      RFC 6350 парсер (склейка строк, параметры, блочные свойства)
│   ├── graph.py      извлечение DAG + топосортировка Кана (+ детектор циклов)
│   ├── ops.py        реестр операций: AVX2 + скалярные эмиттеры
│   ├── jit.py        граф → слитый C → gcc → ctypes; режим kernel-body
│   ├── astvec.py     Python-выражение → AVX2 с выносом констант
│   └── kernels/      встроенные .vcf: pt100, hysteresis, fused, graph
├── tests/            24 pytest-кейса: парсер, граф, численная сверка JIT vs numpy
├── examples/run_kernels.py
├── notebooks/        исходный исследовательский ноутбук (Colab)
└── .github/workflows/ci.yml
```

## Идеи приветствуются

- Графы `RELATED` *между несколькими файлами* (vCard-«линковщик»).
- Бэкенды AVX-512 / NEON за тем же реестром.
- Фиксированные индустриальные операции с политикой округления на карточку.
- Полноценный `vcard-cpu link`, сливающий библиотеки карточек с правилами
  имён.

## Смежные проекты

- [Halide](https://halide-lang.org), [TVM](https://tvm.apache.org) —
  серьёзные версии схемы «опиши вычисления — скомпилируй расписание».
- ONNX — графовый обмен для ML; vcard-cpu — графовый обмен для
  *индустриальных* ядер с чувством юмора.

## Лицензия

Apache-2.0 — см. [LICENSE](LICENSE).
