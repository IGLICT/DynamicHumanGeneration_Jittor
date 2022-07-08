import jittor as jt
jt.flags.use_cuda = 1

# convert texture (tensor [bs,24x3,tex_size,tex_size]) and UV [bs,48,h,w] && Prob [bs,25,h,w] to image  
# def texture2image(textureIm, UVs, Probs, selNUM=None):
#     h, w = UVs.shape[-2:]
#     bs = UVs.shape[0]
#     if (textureIm.shape[0] != bs):
#         textureIm = textureIm.expand(bs,-1,-1,-1)
#     # device = textureIm.device
#     # partNUM = textureIm.shape[1] // 3
#     partNUM = 24
#     chanNUM = textureIm.shape[1] // partNUM
#     if selNUM is None:
#         selNUM = chanNUM
#     # generated_img = jt.zeros([bs, selNUM, h, w]).to(device) # [bs,3,h,w]
#     generated_img = jt.zeros([bs, selNUM, h, w]) # [bs,3,h,w]

#     Probs = jt.nn.softmax(Probs, dim=1)

#     for partID in range(1, partNUM+1):
#         texture = textureIm[:,(partID-1)*chanNUM:(partID-1)*chanNUM+selNUM,:,:] # [bs,3,tex_size,tex_size]
#         uv = UVs[:,(partID-1)*2:partID*2,:,:].permute(0,2,3,1) # [bs,h,w,2]
#         img = jt.nn.grid_sample(texture, uv) # [bs,3,h,w]
#         prob = Probs[:,partID,:,:].unsqueeze(1) # [bs,1,h,w]
#         generated_img += img * prob # [bs,3,h,w]
#         # generated_img += img # [bs,3,h,w]
#     return generated_img
    

def texture2image(textureIm, UVs, Probs, selNUM=None):
    h, w = UVs.shape[-2:]
    bs = UVs.shape[0]
    if (textureIm.shape[0] != bs):
        textureIm = textureIm.expand(bs,-1,-1,-1)
    partNUM = 24
    chanNUM = textureIm.shape[1] // partNUM
    if selNUM is None:
        selNUM = chanNUM

    Probs = jt.nn.softmax(Probs, dim=1)
    # rgb_data = []
    rgb_data = [[] for _ in range(chanNUM)]
    # r = []
    # g = []
    # b = []
    for partID in range(1, partNUM+1):
        texture = textureIm[:,(partID-1)*chanNUM:(partID-1)*chanNUM+selNUM,:,:] # [bs,3,tex_size,tex_size]
        uv = UVs[:,(partID-1)*2:partID*2,:,:].permute(0,2,3,1) # [bs,h,w,2]
        img = jt.nn.grid_sample(texture, uv) # [bs,3,h,w]
        prob = Probs[:,partID,:,:].unsqueeze(1) # [bs,1,h,w]
        data = img * prob  # [bs,3,h,w]
        # rgb_data.append(data)
        for ii in range(chanNUM):
            rgb_data[ii].append(data[:,ii:ii+1,:,:])
        # rgb_data.append([data[:,x:x+1,:,:] for x in range(chanNUM)])  # [[bs,1,h,w]*3] * 24
        # r.append(data[:,0:1,:,:])
        # g.append(data[:,1:2,:,:])
        # b.append(data[:,2:3,:,:])
        # generated_img += img * prob # [bs,3,h,w]

    # r = jt.sum(jt.concat(r,1),1, keepdims=True)
    # g = jt.sum(jt.concat(g,1),1, keepdims=True)
    # b = jt.sum(jt.concat(b,1),1, keepdims=True)
    # rgb_data = jt.stack(rgb_data).permute(1,0,2,3,4,5)  # [[bs,1,h,w]*3] * 24

    # rgb_data = map(sum, rgb_data)
    rgb_data = [sum(x) for x in rgb_data]
    
    generated_img = jt.concat(rgb_data,1)
    # generated_img = sum(rgb_data)
    return generated_img


textureIm = jt.rand(1,72,200,200)
UVs = jt.rand(1,48,512,512)
Probs = jt.rand(1,25,512,512)

img = texture2image(textureIm, UVs, Probs)
print(img.shape)

grad = jt.grad(img, UVs)
print(grad.max(), grad.min(), grad.shape)

grad = jt.grad(img, textureIm)
print(grad.max(), grad.min(), grad.shape)

