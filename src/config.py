config_base = dict(
    d_model=768,
    depth=12,
    n_heads=12,
    expansion=4,
    dropout=.1,
)

config_large = dict(
    d_model=1024,
    depth=24,
    n_heads=16,
    expansion=4,
    dropout=.1,
)

config_huge = dict(
    d_model=1280,
    depth=32,
    n_heads=16,
    expansion=4,
    dropout=.1,
)
