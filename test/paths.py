import os

def local_env_settings():
    settings = EnvSettings()
    settings.davis_dir = ''
    settings.dtb70_path = '/data/yakun/data/DTB70'
    settings.got10k_lmdb_path = '/data/yakun/data/got10k_lmdb'
    settings.got10k_path = '/data/yakun/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/yakun/MultiTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/data/yakun/MultiTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/yakun/data/lasot_lmdb'
    settings.lasot_path = '/data/yakun/data/lasot'
    settings.network_path = '/data/yakun/MultiTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/yakun/MultiTrack/data/nfs'
    settings.otb_path = '/data/yakun/data/otb'
    settings.prj_dir = '/data/yakun/MultiTrack'
    settings.result_plot_path = '/data/yakun/MultiTrack/output/result_plots'
    settings.results_path = '/data/yakun/MultiTrack/output/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/yakun/MultiTrack/output'
    settings.segmentation_path = '/data/yakun/MultiTrack/output/test/segmentation_results'
    settings.tc128_path = '/data/yakun/MultiTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/yakun/MultiTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/yakun/data/trackingnet'
    settings.uav123_10fps_path = '/data/yakun/data/uav123_10fps_path'
    settings.uav123_path = '/data/yakun/data/uav123_path'
    settings.uav_path = '/data/yakun/data/uav'
    settings.uavdt_path = '/data/yakun/data/UAVDT'
    settings.visdrone2018_path = '/data/yakun/data/VisDrone2018'
    settings.vot18_path = '/data/yakun/data/vot2018'
    settings.vot22_path = '/data/yakun/data/vot2022'
    settings.vot_path = '/data/yakun/data/VOT2019'
    settings.threemdot_test_path = '/data/yakun/data/ThreeMDOT/threetest'
    settings.youtubevos_dir = ''

    return settings


class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.results_path = '{}/../output/tracking_results/'.format(test_path)
        self.segmentation_path = '{}/segmentation_results/'.format(test_path)
        self.network_path = '{}/networks/'.format(test_path)
        self.result_plot_path = '{}/result_plots/'.format(test_path)
        self.otb_path = ''
        self.nfs_path = ''
        self.uav_path = ''
        self.tpl_path = ''
        self.vot_path = ''
        self.got10k_path = ''
        self.lasot_path = ''
        self.trackingnet_path = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.dtb70_path = ''
        self.uavdt_path = ''
        self.visdrone2018_path = ''
        self.uav123_path = ''
        self.uav123_10fps_path = ''

        self.got_packed_results_path = ''
        self.got_reports_path = ''
        self.tn_packed_results_path = ''