import argparse
from utils import *
from tensorflow.keras import optimizers

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', type=str, default='')
    parser.add_argument('--train_landmark_dir', type=str, default='')
    parser.add_argument('--test_img_dir', type=str, default='')
    parser.add_argument('--test_landmark_dir', type=str, default='')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model', type=str, default='PCLGAN')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='momentum of adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='momentum of adam')
    opt, _ = parser.parse_known_args()
    return opt
  
  
def main():
  opt = parse_opt()
  model, params = load_model(opt)
  ds_train, ds_val = build_dataset(opt)
  model.compile(
      optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
      optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
      optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
      optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)
  )

  ckpt_dir = f"{opt.ckpt_dir}/{opt.model}/{params}"
  if os.path.exists(ckpt_dir):
      model.load_weights(tf.train.latest_checkpoint(ckpt_dir))

  #sampling 
  for (x, m) in ds_val.take(1):
    b, l, h, w, c = x.shape
    x_prev, m_prev = x[:, :l, ...], m[:, :l, ...]
    x_next, m_next = x[:, 1:, ...], m[:, 1:, ...]
  
  callbacks = set_callbacks(opt, params, x_prev, m_next, val_ds=ds_val)
  
  #train
  model.fit(
      x=ds_train,
      validation_data=ds_val,
      epochs=opt.num_epochs,
      callbacks=callbacks
  )
  
if __name__ == '__main__':
    main()
    
    
