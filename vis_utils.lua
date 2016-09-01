local vis_utils = {}


function vis_utils.colorsToRGB(c)
--[[ Returns the RGB value of common colors (from MATLAB).

Arguments:
  c: color name, either single char or common name string.

Returns:
  rgb: table containing R, G, B values.
--]]
  local rgb
  if     c == 'r' or c == 'red'     then rgb = {255, 0, 0}
  elseif c == 'g' or c == 'green'   then rgb = {0, 255, 0}
  elseif c == 'b' or c == 'blue'    then rgb = {0, 0, 255}
  elseif c == 'y' or c == 'yellow'  then rgb = {255, 255, 0}
  elseif c == 'm' or c == 'magenta' then rgb = {255, 0, 255}
  elseif c == 'c' or c == 'cyan'    then rgb = {0, 255, 255}
  elseif c == 'w' or c == 'white'   then rgb = {255, 255, 255}  
  elseif c == 'k' or c == 'black'   then rgb = {0, 0, 0}
  else error('Invalid color name '.. c)
  end
  return rgb
end


function vis_utils.lineImageFromMask(im, masks, colors)
--[[ Displays lines on image.

Inputs:
  im: Image of size 3xHxW and channel range [0, 255], or filepath string.
  masks: Binary lines masks of size NxHxW, with >= 0 values for line pixels.
  colors: Optional, single char 'r' (same color for all lines) 
   or table of size N (one color per line).

Returns:
  line_im: Image of size 3xHxW with colored lines.
--]]
  colors = colors or 'r'
  local num_masks = masks:size(1)
  if type(colors) ~= 'table' then
    local color = colors
    colors = {}
    for i = 1, num_masks do
      colors[#colors+1] = color
    end
  end
  assert(#colors == masks:size(1), 
    'Need one color per line as input, or one color for all lines')

  if type(im) == 'string' then
    im = image.load(im)
    -- Resize image.
    local H, W = masks:size(2), masks:size(3)
    im = image.scale(im, W, H):float():mul(255)
  end

  -- Clone input image to prevent overwriting the image...
  local line_im = im:clone()

  -- Add in color for each mask.
  for i = 1, num_masks do
    -- TODO: Figure out how to do logical indexing over channels.
    -- Until then use this loop. 
    local line_color = vis_utils.colorsToRGB(colors[i])
    for c = 1, 3 do
      local channel = line_im[{c, {}}]
      channel[masks[i]:gt(0)] = line_color[c]
    end
  end

  return line_im
end


function vis_utils.tile(imgs, h, w, opt)
--[[ Takes in 4D tensor of images (N, channel, height, width), and tiles them
into a grid of size hxw. If h and w are not provided then does a square tiling.
--]]
  assert(imgs:dim() == 4)

  opt = opt or {}
  margin = opt.margin or 1

  local N = imgs:size(1)
  if h == nil or w == nil then
    h = torch.floor(torch.sqrt(N))
    w = h
    if h * w < N then
      w = w + 1
    end
  end
  assert(h * w >= N) 

  local H, W = imgs:size(3), imgs:size(4)
  local out = torch.ones(imgs:size(2), h * imgs:size(3) + (h - 1) * margin, 
    w * imgs:size(4) + (w - 1) * margin)
  for index = 1, N do
    local i = torch.floor((index-1) / w)
    local j = (index % w) - 1
    local i_start = i * (H + margin) + 1
    local i_end = i_start + H - margin
    local j_start = j * (W + margin) + 1
    local j_end = j_start + W - margin
    out[{{}, {i_start, i_end}, {j_start, j_end}}] = imgs[index]
  end

  return out
end


return vis_utils