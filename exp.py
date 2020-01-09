import hydra


@hydra.main(config_path="conf/config.yaml")
def my_app(cfg):
    connection = hydra.utils.instantiate(cfg.db)
    connection.connect()


if __name__ == "__main__":
    my_app()
