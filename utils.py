import os
import tensorflow as tf
from sklearn.model_selection import train_test_split as ttp
from models import PCLGAN
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import yaml
from metrics import metrics
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_model(opt):
    config = get_config(f'./configs/{opt.model}.yaml')
    if opt.model == 'PCLGAN':
        model = PCLGAN.PCLGAN(config)
        params = f"{config['loss_type']}_{config['tau']}" \
                 f"_{config['num_patches']}_{config['use_identity']}_{config['use_perceptual']}"
    return model, params


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_image(pth, num_channels, opt):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=num_channels)
    image = tf.cast(tf.image.resize(image, (opt.image_size, opt.image_size)), 'float32')
    return (image-127.5)/127.5

def build_tf_dataset(img_seq, landmark_seq, opt, train=True):
    ds_img = tf.data.Dataset.from_generator(lambda: img_seq, 
                                         tf.as_dtype('float32'),
                                         tf.TensorShape([None, opt.image_size, opt.image_size, 3]))
    
    ds_landmark = tf.data.Dataset.from_generator(lambda: img_landmark, 
                                         tf.as_dtype('float32'),
                                         tf.TensorShape([None, opt.image_size, opt.image_size, 1]))
    
    if train:
        ds = tf.data.Dataset.zip((ds_img, ds_landmark)).\
          shuffle(256).prefetch(AUTOTUNE)
    else:
        ds = tf.data.Dataset.zip((ds_img, ds_landmark)).prefetch(AUTOTUNE)
    return ds

def build_seq_list(dir_, num_channels, opt):
    seq_list = list(map(lambda x: f'{dir_}/{x}', sorted(os.listdir(dir_))))
    seq_list=list(map(lambda x: map(lambda y: get_image(f'{x}/{y}', num_channels, opt), sorted(os.listdir(x))), seq_list))
    seq_list=list(map(lambda x: tf.concat([list(x)], axis=0), seq_list))
    return seq_list

def build_dataset(opt):
  #imgs
  train_img = build_seq_list(opt.train_img_dir, 3, opt)
  train_landmark = build_seq_list(opt.train_landmark_dir, 1, opt)
  test_img = build_seq_list(opt.test_img_dir, 3, opt)
  test_landmark = build_seq_list(opt.test_landmark_dir, 1, opt)
  
  #tf-dataset
  ds_train = build_tf_dataset(train_img, train_landmark, opt)
  ds_val = build_tf_dataset(test_img, test_landmark, opt, False)
  return ds_train, ds_val

def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[0:RY, 0] = 1
    colorwheel[0:RY, 1] = np.arange(0, 1, 1. / RY)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = np.arange(1, 0, -1. / YG)
    colorwheel[col:col + YG, 1] = 1
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 1
    colorwheel[col:col + GC, 2] = np.arange(0, 1, 1. / GC)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = np.arange(1, 0, -1. / CB)
    colorwheel[col:col + CB, 2] = 1
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 1
    colorwheel[col:col + BM, 0] = np.arange(0, 1, 1. / BM)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = np.arange(1, 0, -1. / MR)
    colorwheel[col:col + MR, 0] = 1

    return colorwheel

def viz_flow(u, v, logscale=True, scaledown=6, output=False):
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    radius = radius / scaledown
    rot = np.arctan2(-v, -u) / np.pi
    fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape + (ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1 - f) * col0 + f * col1

        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        # out of range
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col)
    return img/255.

###Callbacks
class VisualizeCallback(callbacks.Callback):
    def __init__(self, x_prev, m_prev, x_next, m_next, opt, params):
        super().__init__()
        self.x_prev, self.m_prev, self.x_next, self.m_next = x_prev, m_prev, x_next, m_next
        self.opt = opt
        self.params_ = params

    def on_epoch_end(self, epoch, logs=None):
        l, h, w, c = self.x_prev.shape

        if self.opt.model == 'PCLGAN':
            x2y, info = self.model.G([self.x_prev, self.m_next])
            x_warped, flow, a, r = info

        fig, ax = plt.subplots(nrows = 6, ncols =len(l), figsize=(16, 16))

        for i in range(l):
            if self.opt.model == 'PCLGAN':
                #prev
                ax[0, i].imshow(self.x_prev[i] * 0.5 + 0.5, cmap='gray')
                ax[0, i].axis('off')
                
                #next
                ax[1, i].imshow(self.x_next[i] * 0.5 + 0.5, cmap='gray')
                ax[1, i].axis('off')
                
                #next generated
                ax[2, i].imshow(x2y[i] * 0.5 + 0.5, cmap='gray')
                ax[2, i].axis('off')
                
                #flow 
                grid_img = viz_flow(flow[i, ..., 0], flow[i, ..., 1])
                ax[3, i].imshow(grid_img)
                ax[3, i].axis('off')
                
                #warped 
                ax[4, i].imshow(x_warped[i] * 0.5 + 0.5, cmap='gray')
                ax[4, i].axis('off')
                
                #attention mask
                ax[4, i].imshow(a[i], cmap='gray')
                ax[4, i].axis('off')
                
                #residual 
                ax[5, i].imshow(r[i] * 0.5 + 0.5, cmap='gray')
                ax[5, i].axis('off')
                        
        dir = f'{self.opt.output_dir}/{self.opt.model}/{self.params_}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/synthesis_{epoch}.jpg')


def set_callbacks(opt, params, x_prev, m_prev, x_next, m_next, val_ds=ds_val):
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}"
    output_dir = f"{opt.output_dir}/{opt.model}"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{ckpt_dir}/{params}/{opt.model}", 
        save_weights_only=True, 
        monitor='val_ms_ssim', 
        save_best_only=True)
    history_callback = callbacks.CSVLogger(f"{output_dir}/{params}.csv", separator=",", append=False)
    visualize_callback = VisualizeCallback(x_prev, m_prev, x_next, m_next, opt, params)
    early_stopping_callback = callbacks.EarlyStopping(monitor='val_ms_ssim', patience=15, restore_best_weights=True)
    metrics_callback = metrics.MetricsCallbacks(val_ds, opt, params)
    return [checkpoint_callback, history_callback, visualize_callback, metrics_callback, early_stopping_callback]
