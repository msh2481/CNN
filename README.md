# Sirius-SPbSU-2022: entry competition
## Что происходило до смены правил
Сначала я не планировал никуда отбираться и отчёт об экспериментах не готовил, поэтому перескажу примерный порядок по памяти.
1. Попробовал `torchvision.models.resnet18` с Adam или SGD и слегка подобранными гиперпараметрами. ~87%
2. Сгладил чёрно-белые точки в исходных данных. ~89%
3. Вставил `nn.Dropout` между всеми слоями, кажется не помогло. 
4. Добавил `torchvision.transforms.AutoAugment` в обучение. ~91%
5. Попробовал делать эту же аугментацию несколько раз при валидации и тестировании чтобы получить что-то вроде ансамбля, не помогло.
6. Добавил все архитектуры из https://arxiv.org/abs/2008.10400, на входе добавил `CenterCrop(28)`, чтобы по возможности не подбирать архитектуру заново. ~92%
7. Попробовал ансамбли из раннее обученных моделей, двух видов: где модели просто голосуют, и где они соединяются дополнительным линейным слоем. ~93%
8. В какой-то момент перешёл на QHAdam, потому что он точно не хуже SGD и Adam при правильно подобранных параметрах, но пока не успел эти параметры перебрать.
9. Попробовал tianshou.PrioritizedReplayBuffer (в оригинальной статье писали, что на MNIST помогает), стало вдруг дико переобучаться несмотря на агументацию.
10. Добавил cutout (вырезание квадратов из изображения), но не успел настроить его до нужной силы регуляризации.
## Собственно отчёт
Заменил `Autoaugment` на отдельные преобразования, добавил параметры к cutout.
По-новой сгладил данные, на этот раз получилось лучше, шума совсем не видно.
Наконец сделал нормальный перебор гиперпараметров, для начала просто случайный поиск, распределения такие (`norm_rnd(mean, std, clip_min, clip_max)`):


    cutout_min = norm_rnd(4, 4, 0, 16)
    bs = 2**randint(5, 10)
    'jitter_brightness': norm_rnd(0, 0.1, 0, 0.5),
    'jitter_contrast': norm_rnd(0, 0.1, 0, 0.5),
    'jitter_saturation': norm_rnd(0, 0.1, 0, 0.5),
    'jitter_hue': norm_rnd(0, 0.1, 0, 0.5),
    'perspective_distortion': norm_rnd(0, 0.1, 0, 1),
    'cutout_count': int(norm_rnd(0, 1, 0, 10)),
    'cutout_min_size': int(cutout_min),
    'cutout_max_size': int(cutout_min * norm_rnd(2, 0.5, 1, 10)),
    'model': choice(['M5()', 'M7()', 'Resnet18(10)']),
    'batch_size': bs,
    'optimizer': 'QHAdam',
    'lr': 10**norm_rnd(-3, 1, -6, -1),
    'wd': 10**norm_rnd(-4, 1, -7, -2),
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': norm_rnd(0.5, 0.2, 0.1, 0.9),
    'nu2': norm_rnd(1, 0.1, 0.8, 1)

Я запустил примерно 50 итераций, по 2 эпохи каждая. Из них я понял, что M5 работает лучше resnet и M7, как минимум в начале. Немного уточнил диапазон lr, теперь я считаю центром 1e-4. А с остальными параметрами, особенно отвечающими за регуляризацию, было мало что понятно, потому что, во-первых, на первых итерациях регуляризация и не нужна, а во-вторых, 2 эпохи это довольно коротко и в результатах слишком много шума из-за разной иннициализации начальных весов. Ну и чтобы перебирать параметры более эффективно, следующий перебор запускал с Optuna. Часть параметров я просто зафиксировал для простоты и в качестве начальной точки взял лучшую модель с предыдущих итераций (точность 93% на валидации).

    cutout_min = trial.suggest_int('cutout_min_size', 2, 8, log=True)
    bs = 64
    params = {
        'jitter_brightness': 0.01,
        'jitter_contrast': 0.01,
        'jitter_saturation': 0.01,
        'jitter_hue': 0.01,
        'perspective_distortion': 0.01,
        'cutout_count': 1,
        'cutout_min_size': cutout_min,
        'cutout_max_size': trial.suggest_int('cutout_max_size', cutout_min + 1, 3 * cutout_min, log=True),
        'model': 'load_from_zoo("M5()_61f1af58_final.p")',
        'batch_size': bs,
        'optimizer': 'QHAdam',
        'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
        'wd': trial.suggest_float('wd', 1e-7, 1e-3, log=True),
        'beta1': 0.9,
        'beta2': 0.999,
        'nu1': trial.suggest_float('nu1', 0.1, 0.9),
        'nu2': 1,
        'epochs': 5,
    }

Здесь наоборот регуляризация (cutout_min_size, в частности) сыграла решающую роль, lr ещё немного уменьшился, для nu1 и wd точного значения снова не получилось найти, зафиксирую пока на 0.6 и 1e-5. Получил модель с ~94% на валидации, теперь попробую дообучать её по 10 эпох, подбирая больше параметров для регуляризации. Параллельно собрал ансамбль из нескольких предыдущих моделей с 93-94%, получилось 95% на валидации и 94.1% на открытых тестах.


    cutout_min = trial.suggest_int('cutout_min_size', 4, 16, log=True)
    bs = 64
    params = {
        'jitter_brightness': trial.suggest_float('jitter_brightness', 0.005, 0.5, log=True),
        'jitter_contrast': trial.suggest_float('jitter_contrast', 0.005, 0.5, log=True),
        'jitter_saturation': trial.suggest_float('jitter_saturation', 0.005, 0.5, log=True),
        'jitter_hue': trial.suggest_float('jitter_hue', 0.005, 0.5, log=True),
        'perspective_distortion': trial.suggest_float('perspective_distortion', 0.005, 1, log=True),
        'cutout_count': trial.suggest_int('cutout_count', 1, 4, log=True),
        'cutout_min_size': cutout_min,
        'cutout_max_size': cutout_min * 2,
        'model': 'load_from_zoo("M5()_61f26677_final.p")',
        'batch_size': bs,
        'plot_interval': (4000 + bs - 1) // bs,
        'train': 'train_v2.bin',
        'use_per': False,
        'val': 'val_v2.bin',
        'test': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': 'QHAdam',
        'lr': 5e-5,
        'wd': 1e-5,
        'beta1': 0.9,
        'beta2': 0.999,
        'nu1': 0.6,
        'nu2': 1,
        'epochs': 10
    }

Точность на валидации заметно повысить не получилось, но зато подобрал какие-то значения для всех аугментаций, теперь зафиксировав их опять поперебираю wd, lr, nu1 по 10 эпох.

    'jitter_brightness': 0.03,
    'jitter_contrast': 0.05,
    'jitter_saturation': 0.27,
    'jitter_hue': 0.03,
    'perspective_distortion': 0.12,
    'cutout_count': 2,
    'cutout_min_size': 6,
    'cutout_max_size': 12,
    'model': 'load_from_zoo("M5()_61f271e4_final.p")',
    'batch_size': 64,
    'optimizer': 'QHAdam',
    'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
    'wd': trial.suggest_float('wd', 1e-6, 1e-4, log=True),
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': trial.suggest_float('nu1', 0.4, 0.8),
    'nu2': 1,
    'epochs': 10

## TODO
https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py
https://github.com/hiyouga/AMP-Regularizer
Stochastic depth.
LR finder.
Другие архитектуры.
Дистилляция чего-нибудь с чем-нибудь.
Сглаживание весов.
Регуляризация через weight decay.
MaxNorm на веса или градиенты.
