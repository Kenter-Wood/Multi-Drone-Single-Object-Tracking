from .lasot import Lasot
from .lasot_lmdb import Lasot_lmdb
from .lasot_for_three import LasotForThree
from .got10k import Got10k
from .got10k_lmdb import Got10k_lmdb
from .got_for_three import GOTForThree
from .threemdot import ThreeMDOT




def names2datasets(name_list: list, settings, image_loader):
    """
    输入：一些数据集的名称
    输出：对应的数据集类
    """
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "THREEMDOT", "THREEMDOT_VAL", "LASOTForThree", "GOTForThree"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "THREEMDOT":
            datasets.append(ThreeMDOT(settings.env.threemdot_dir, split='train', image_loader=image_loader))
        if name == "THREEMDOT_VAL":
            datasets.append(ThreeMDOT(settings.env.threemdot_dir, split='val', image_loader=image_loader))
        if name == "LASOTForThree":
            datasets.append(LasotForThree(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOTForThree":
            datasets.append(GOTForThree(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
    return datasets