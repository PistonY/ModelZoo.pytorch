# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)


# init DALI
try:
    import nvidia.dali as dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
except ImportError:
    print('DALI is not available')


class TrainPipe(dali.pipeline.Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, crop, color_jit=0.4, use_cpu=False):
        super(TrainPipe, self).__init__(batch_size, num_threads, device_id)
        dali_device = 'cpu' if use_cpu else 'gpu'
        decoder_device = 'cpu' if use_cpu else 'mixed'

        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

        self.input = dali.ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=1,
                                         shuffle_after_epoch=True)

        self.decode = dali.ops.ImageDecoderRandomCrop(device=decoder_device, output_type=dali.types.RGB,
                                                      device_memory_padding=device_memory_padding,
                                                      host_memory_padding=host_memory_padding,
                                                      num_attempts=100)

        self.res = dali.ops.Resize(device=dali_device, resize_x=crop, resize_y=crop,
                                   interp_type=dali.types.INTERP_TRIANGULAR)

        self.bri = dali.ops.Brightness(device=dali_device)
        self.con = dali.ops.Contrast(device=dali_device)
        self.sat = dali.ops.Saturation(device=dali_device)

        self.cmnp = dali.ops.CropMirrorNormalize(device=dali_device, output_dtype=dali.types.FLOAT,
                                                 output_layout=dali.types.NCHW,
                                                 crop=(crop, crop), image_type=dali.types.RGB,
                                                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.coin = dali.ops.CoinFlip(probability=0.5)
        self.uniform = dali.ops.Uniform(range=(max(0., 1 - color_jit), 1 + color_jit))

    def define_graph(self):
        imgs, labels = self.input(name='Reader')
        imgs = self.decode(imgs)
        imgs = self.res(imgs)
        imgs = self.bri(imgs, brightness=self.uniform())
        imgs = self.con(imgs, contrast=self.uniform())
        imgs = self.sat(imgs, saturation=self.uniform())
        imgs = self.cmnp(imgs, mirror=self.coin())
        return imgs.gpu(), labels.gpu()


class ValPipe(dali.pipeline.Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, resize, crop, use_cpu=False):
        super(ValPipe, self).__init__(batch_size, num_threads, device_id)
        dali_device = 'cpu' if use_cpu else 'gpu'
        decoder_device = 'cpu' if use_cpu else 'mixed'

        self.input = dali.ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=1,
                                         shuffle_after_epoch=True)
        self.decode = dali.ops.ImageDecoder(device=decoder_device, output_type=dali.types.RGB)
        self.res = dali.ops.Resize(device=dali_device, resize_shorter=resize,
                                   interp_type=dali.types.INTERP_TRIANGULAR)
        self.cmnp = dali.ops.CropMirrorNormalize(device=dali_device, output_dtype=dali.types.FLOAT,
                                                 output_layout=dali.types.NCHW,
                                                 crop=(crop, crop), image_type=dali.types.RGB,
                                                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        imgs, labels = self.input(name='Reader')
        imgs = self.decode(imgs)
        imgs = self.res(imgs)
        imgs = self.cmnp(imgs)
        return imgs.gpu(), labels.gpu()
