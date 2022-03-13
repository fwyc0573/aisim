
from tf_profile import tf_profile


def test_output_csv():
    # tset  tf_profile.output_csv() function
    filename = 'output_csv.csv'
    path = './tests/output'
    csv_list = [[.1, .2, .3, .4],
                [2, 3, 4, 5],
                [4, 5, 6, 7]]
    tf_profile.output_csv(filename, csv_list, path=path, scale=10)


def test_run_model():

    # test  tf_profile.run_model() function
    tf_profile.run_model('alexnet', loss=True, output='tests/output')

    # test profile
    tf_profile.run_model('resnet50', profile=True, output='tests/output')

    # test timeline
    tf_profile.run_model('vgg16', timeline=True, output='tests/output')

    # test multigpus
    tf_profile.run_model('inception3', timeline=True, gpu_num=4, session=3,
                         step=10, output='tests/output')

    # test horovod
    tf_profile.run_model('vgg16', gpu_num=4, horovod=True,
                         output='tests/output')
