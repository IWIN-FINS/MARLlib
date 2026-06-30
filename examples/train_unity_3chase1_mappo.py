from marllib import marl


def main():
    env = marl.make_env(
        environment_name="unity_3chase1",
        map_name="3Chase1",
        no_graphics=False,
        env_base_port=7200,
    )
    algo = marl.algos.mappo(hyperparam_source="test")
    model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "64-64"})
    algo.fit(
        env,
        model,
        stop={"training_iteration": 1},
        local_mode=True,
        num_gpus=0,
        num_workers=0,
        share_policy="individual",
        checkpoint_end=False,
    )


if __name__ == "__main__":
    main()
