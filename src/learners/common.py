import torch.nn as nn
import learners.models as M
activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
}


def build_gcnpolicy(conf, cons_dim=None, vars_dim=None, edges_dim=1):
    model = M.GCNPolicy(
        emb_size = int(conf["embedding_size"]),
        n_cons_f = cons_dim,
        n_var_f = vars_dim,
        n_edge_f = edges_dim,
        n_layers = int(conf["gcnpolicy_n_layers"]),
        aggregation = conf["gcnpolicy_aggregation"],
        dev = conf["dev_mode"],
        bip_layer_anticipate_mp = conf["bip_layer_anticipate_mp"],
        bip_conv_order = conf["bip_conv_order"],
        bip_conv_append_activation = conf["bip_conv_append_activation"],
        force_bias_value = conf.get("force_bias_value", False),
    )
    return model
#<

def build_gcnpolicy_ext(conf, cons_dim=None, vars_dim=None, edges_dim=1):
    model = M.GCNPolicy2(
        emb_size = int(conf["embedding_size"]),
        n_cons_f = cons_dim,
        n_var_f = vars_dim,
        n_edge_f = edges_dim,
        n_layers = int(conf["gcnpolicy_n_layers"]),
        aggregation = conf["gcnpolicy_aggregation"],
        dev = conf["dev_mode"],
        bip_layer_anticipate_mp = conf["bip_layer_anticipate_mp"],
        bip_conv_order = conf["bip_conv_order"],
        bip_conv_append_activation = conf["bip_conv_append_activation"],
        force_bias_value = conf.get("force_bias_value", False),
    )
    return model
#<




def build_model(conf, cons_dim=None, vars_dim=None, edges_dim=1):
    model_name = conf.get("model_name", "gcnpolicy")
    model = None
    if model_name == "gcnpolicy":
        model = build_gcnpolicy(conf, cons_dim, vars_dim, edges_dim)
    elif model_name == "gcnpolicy2":
        print("Building ext model")
        model = build_gcnpolicy_ext(conf, cons_dim, vars_dim, edges_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
#<