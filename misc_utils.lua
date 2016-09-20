local cjson = require 'cjson'
require 'gnuplot'


local utils = {}


function utils.makePlotTensor(results_history)
  local num_vals = utils.count_keys(results_history)
  local x = torch.Tensor(num_vals)
  local y = torch.Tensor(num_vals)
  local i = 1
  for k, v in pairs(results_history) do
    x[i] = k
    y[i] = v
    i = i + 1
  end
  -- Sort by x axis.
  x_val, index = torch.sort(x)
  y_val = y:index(1, index)

  return x_val, y_val
end


function utils.plotAccuracy(results_history, filename)
  x_val, y_val = utils.makePlotTensor(results_history)

  gnuplot.pngfigure(filename .. '.png')
  -- gnuplot.plot('Accuracy', x_val, y_val, ' using 1:2:(sprintf("(%d, %d)", $1, $2)) with labels ')
  gnuplot.plot('Accuracy', x_val, y_val)
  gnuplot.raw('using 0:2:1 with labels offset 0,char 1')
  gnuplot.xlabel('Iteration')
  gnuplot.ylabel('Accuracy')
  gnuplot.plotflush()
end


function utils.getLearningRate(opt, iter)
  if opt.decay_type == 'fixed' then
    return opt.base_lr
  elseif opt.decay_type == 'step' then
    return opt.base_lr * math.pow(opt.gamma, math.floor(iter / opt.step))
  elseif opt.decay == 'exp' then
    return opt.base_lr * math.pow(opt.gamma, iter)
  end
end


function utils.findArgMaxBatch(vals, num_in_batch)
--[[ 
values is of size num_in_batch*N x 1.
Finds the index with max value in every num_in_batch x 1 vector.
--]]
  local values = vals:clone()
  values = torch.squeeze(values)

  assert(values:size(1) % num_in_batch == 0)
  local num_batches = values:size(1) / num_in_batch
  local output = torch.Tensor(num_batches):zero()

  for i = 1, num_batches do
    local start_index = (i-1)*num_in_batch + 1
    local end_index = start_index + num_in_batch - 1 
    local batch = values[{{start_index, end_index}}]
    local _, max_ind = torch.max(batch, 1)
    output[i] = max_ind[1]
  end

  return output
end


function utils.deepcopy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in next, orig, nil do
          copy[utils.deepcopy(orig_key)] = utils.deepcopy(orig_value)
      end
      setmetatable(copy, utils.deepcopy(getmetatable(orig)))
  else -- number, string, boolean, etc
      copy = orig
  end
  return copy
end


function utils.find_key_of_value_in_table(table, value)
--[[ Finds simple value (number, string, boolean, etc.) and returns key if 
found in table. This returns the first occurrence.
--]]
  for k, v in pairs(table) do
    if v == value then
      return k
    end
  end
  return nil
end


--[[
Utility function to set up GPU imports and pick datatype based on commmand line
arguments.

Inputs:
- gpu: Index of GPU requested on the command line; zero-indexed. gpu < 0 means
  CPU-only mode. If gpu >= 0 then we will import cutorch and cunn and set the
  current device using cutorch.
- use_cudnn: Whether cuDNN was requested on the command line; either 0 or 1.
  We will import cudnn if gpu >= 0 and use_cudnn == 1.

Returns:
- dtype: torch datatype that should be used based on the arguments
- actual_use_cudnn: Whether cuDNN should actually be used; this is equal to
  gpu >= 0 and use_cudnn == 1.
--]]
function utils.setup_gpus(gpu, use_cudnn)
  local dtype = 'torch.FloatTensor'
  local actual_use_cudnn = false
  if gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpu + 1)
    dtype = 'torch.CudaTensor'
    if use_cudnn == 1 then
      require 'cudnn'
      actual_use_cudnn = true
    end
  end
  return dtype, actual_use_cudnn
end


--[[
get a table of string->float of losses and put them all together 
into a printable human readable string.
--]]
function utils.build_loss_string(losses)
  local x = ''
  for k,v in pairs(losses) do
    if k ~= 'total_loss' then
      x = x .. string.format('%s: %.3f, ', k, v)
    end
  end
  x = x .. string.format(' [total: %.3f]', losses.total_loss)
  return x
end

--[[
get a table of string->float of timings and produce a nice string report
--]]
function utils.build_timing_string(timings)
  local x = ''  
  -- todo: maybe print in sorted order?
  for k,v in pairs(timings) do
    x = x .. string.format('timing %s: %.3fms', k, v*1000) .. '\n'
  end
  return x
end

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

-- Check to make sure a required key is present
function utils.ensureopt(opt, key)
  if opt == nil or opt[key] == nil then
    error('error: required key ' .. key .. ' was not provided.')
  end
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

-- table vals to string
function utils.table_v_to_str(tbl)
  str = ''
  for k, v in pairs(tbl) do
    str = str .. tostring(v) .. ', '
  end
  return str:sub(1, (#str)-2)
end
-- table to string
function utils.table_to_str(tbl)
  str = ''
  for k, v in pairs(tbl) do
    str = str .. tostring(k) .. '=' .. tostring(v) .. ', '
  end
  return str:sub(1, (#str)-2)
end

-- From https://github.com/torch/xlua/blob/master/init.lua.
function utils.split_string(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
         table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

-- Stash global statistics here.
-- Since loading files with require caches and does not reload the same file,
-- all places that require 'utils' will have access to this table.
-- Loading with dofile will not cache, and will therefore give a fresh copy.
utils.__GLOBAL_STATS__ = {}

return utils