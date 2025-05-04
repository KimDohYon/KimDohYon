from collections import namedtuple

# genotype를 표현하는 구조 (normal/reduce 셀의 ops와 각 노드 연결 정보)
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# (탐색 후 얻은 최적 구조를 여기에 기재합니다)
# 아래는 예시 값이며, 실제 탐색 결과에 따라 대체해야 합니다.
NASGenotype = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('skip_connect', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 2),
        ('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2), ('max_pool_3x3', 3)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3', 0), ('skip_connect', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 2),
        ('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2), ('max_pool_3x3', 3)
    ],
    reduce_concat=[2, 3, 4, 5]
)

