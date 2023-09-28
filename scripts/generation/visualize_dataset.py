import imageio
import h5py

def data_to_video(file, T=1000):
    suffix = file.split(".")[0]
    f = h5py.File(file, 'r')
    first_imgs = f['images'][:T]
    imageio.mimwrite(f"first_imgs_{suffix}.mp4", first_imgs, fps=100)

if __name__ == '__main__':
   
    files = ["pointmass_original.hdf5",
             "pointmass_obscure_1.hdf5",
             "pointmass_obscure_3.hdf5",
             "pointmass_fpv.hdf5"]
    files = ["pointmass0_hidden.hdf5",]
    
    for file in files:
        data_to_video(file)
    