def autosharding_runner(model_name, checkpoint):
    config_sequence_length = 128 if model_name == "gpt2" else 96
    config_batch_size = 10 * 1024

    if model_name == "gpt2":
        config = GPT2Config()
        config.n_layer = 1
        model = GPT2Model(config)
    else:
        config = BertConfig()
        config.num_hidden_layers = 1
        config.hidden_size = 96
        config.intermediate_size = 384
        config._attn_implementation = "eager"
        model = BertModel(config)

    mg = MaseGraph(model)
    pipeline = AutoPipelineForDistributedInference()

    mg = pipeline(
        mg,
        pass_args={
            "report_graph_analysis_pass": {
                "file_name": f"{checkpoint.replace('/', '-')}-graph.txt"
            },
            "add_common_metadata_analysis_pass": {
                "dummy_in": {
                    "input_ids": torch.randint(0, 10, (1, config_sequence_length)),
                    # "input_ids": torch.randn((1, config_sequence_length, config.hidden_size)),
                },
                "add_value": False,
            },
            "autosharding_analysis_pass": {
                "mesh_shape": MESH_SHAPE,
                "inter_node_bandwidth": 10e9,
                "intra_node_bandwidth": 100e9,
            },
            "resharding_transform_pass": {
                "module_map": "self/autosharding_analysis_pass",  # output of autosharding_analysis_pass is directed to resharding_transform_pass
                "device_mesh": DEVICE_MESH,
            },
        },
    )

    mg.draw()
