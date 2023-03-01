from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import math
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# add more samples here later
samples = [
    '../data/230213_f459_ubi_flk_6dpf.zarr'
]

# how many samples to combine for one batch
batch_size = 8

# shape of a sample in voxels
input_shape = gp.Coordinate((34, 140, 140))

# size of a voxel in nm
voxel_size = gp.Coordinate((1000, 325, 325))
mask_voxel_size = gp.Coordinate((2 * 1000, 8 * 325, 8 * 325))

# how to normalize raw/labels to be (roughly) between 0 and 1
normalize_raw = 1.0 / 22780.0
normalize_labels = 1.0 / 34949.0

class AddMaskRequest(gp.BatchFilter):

    def __init__(self, mask, voxel_size, reference):

        self.mask = mask
        self.voxel_size = voxel_size
        self.reference = reference

    def prepare(self, request):

        reference_roi = request[self.reference].roi
        mask_request_roi = reference_roi.snap_to_grid(
            self.voxel_size,
            mode='grow')

        deps = gp.BatchRequest()
        deps[self.mask] = gp.ArraySpec(roi=mask_request_roi)

        return deps

    def process(self, batch, request):
        pass


def create_model():

    unet = UNet(
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=4,
        downsample_factors=[
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2)
        ],
        activation='ReLU',
        constant_upsample=True,
        padding='valid')

    model = torch.nn.Sequential(
        unet,
        ConvPass(12, 1, [(1, 1, 1)], activation=None),
        torch.nn.Sigmoid())

    return model


def train_until(iterations):

    model = create_model()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4)

    # declare all arrays that we will use in this pipeline

    input_size = input_shape * voxel_size
    output_shape = gp.Coordinate(model(torch.zeros((1, 1,) + input_shape)).shape[2:])
    output_size = output_shape * voxel_size

    logger.info("Input shape : %s", input_shape)
    logger.info("Output shape: %s", output_shape)
    logger.info("Input size : %s", input_size)
    logger.info("Output size: %s", output_size)

    raw = gp.ArrayKey('RAW')
    mask = gp.ArrayKey('MASK')
    labels = gp.ArrayKey('LABELS')
    prediction = gp.ArrayKey('PREDICTION')

    # define a request (i.e., a single sample in a batch)
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(prediction, output_size)

    # create a zarr source, one for each sample

    sources = tuple(
        gp.ZarrSource(
            sample,
            {
                raw: 'raw',
                mask: 'mask',
                labels: 'labels'
            }
        )
        for sample in samples
    )

    # assemble the pipeline

    pipeline = (

        sources +

        # pick a random sample
        gp.RandomProvider() +

        # pick a random volume covered by the mask
        gp.RandomLocation(min_masked=0.5, mask=mask) +

        # add a request for the mask to ensure that we don't have background in
        # the sample
        AddMaskRequest(mask, mask_voxel_size, labels) +

        # scale raw and labels
        gp.Normalize(raw, factor=normalize_raw) +
        gp.Normalize(labels, factor=normalize_labels) +

        # rotate and deform a little
        gp.ElasticAugment(
            control_point_spacing=[4, 40, 40],
            jitter_sigma=[0, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            subsample=8) +

        # mirror and flip (the latter only in x and y)
        gp.SimpleAugment(transpose_only=[1, 2]) +

        # change intensity
        gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +

        # make things faster :)
        gp.PreCache(
            cache_size=40,
            num_workers=10) +

        # create a "channel" dimension
        gp.Stack(1) +

        # create batches of multiple samples
        gp.Stack(batch_size) +

        # and train!
        gp.torch.Train(
            model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            loss_inputs={
                0: prediction,
                1: labels
            },
            outputs={
                0: prediction
            },
            save_every=10000,
            log_dir='log') +

        # log some batches every once in a while
        gp.Snapshot({
                raw: 'raw',
                labels: 'labels',
                prediction: 'prediction'
            },
            every=100,
            output_filename='batch_{iteration}.zarr') +

        # see where we spend most time
        gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):
        for _ in range(iterations):
            pipeline.request_batch(request)


if __name__ == '__main__':

    train_until(1000)
