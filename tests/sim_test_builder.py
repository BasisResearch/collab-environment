import yaml


from collab_env.data.file_utils import get_project_root, expand_path


def build_config_file(config_input_file=None, config_output_file=None, options=None):
    config = yaml.safe_load(open(expand_path(config_input_file, get_project_root())))
    print(f"config: \n{config}")

    for outer_key, outer_value in options.items():
        for inner_key, inner_value in outer_value.items():
            config[outer_key][inner_key] = inner_value
    print(f"updated config:\n{config}")

    output_path = expand_path(config_output_file, get_project_root())
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    config = yaml.safe_load(open(expand_path(config_output_file, get_project_root())))
    print(f"new config: \n{config}")


if __name__ == "__main__":
    test_case = {"simulator": {"walking": False}}
    build_config_file(
        "collab_env/sim/boids/config.yaml",
        "tests/sim_built_config.yaml",
        options=test_case,
    )
