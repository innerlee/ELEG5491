using HDF5
using Images
using ImageView

file = "runs/net1/snapshots/iter.best.parrots"
arg = "conv1.w@value"

data = h5read(file, arg)

l, u = minimum(data), maximum(data)

data = (data .- l) ./ (u - l)

data = permutedims(data, [3,1,2,4])

for i=1:size(data,4)
    imshow(colorview(RGB, data[:,:,:,i]))
end
