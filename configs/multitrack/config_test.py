from utils.params import TrackerParams
import os
from test.settings import env_settings
from configs.multitrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'scripts/multitrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "test_models/multitrack/%s/ThreeTrack_ep%04d.pth.tar" %
                                    (yaml_name, cfg.TEST.EPOCH))
    # params.checkpoint = os.path.join(save_dir, "checkpoints/test/mythreetrack/%s/ThreeMDOT.pth.tar" %
                                      # yaml_name)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params