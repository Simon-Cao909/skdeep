from skdeep.tools.building.quick_parser import parse_eqn

print(parse_eqn("E_t + ∇⋅(EV) + ∇⋅(pV) + (1/RP)∇⋅q = \
                (1/R)(a⋅V)_x + \
                (1/R)(b⋅V)_y + \
                (1/R)(c1 + V)_z"))