local wordemb = require 'wordemb'


local embedding_utils = {}


function embedding_utils.init(filepath)
  local w2v = wordemb.load_word2vec_bin(filepath)
  return w2v
end


function embedding_utils.word2vec(w2v, sentence, opts)
  opts = opts or {}
  local average = opts.average or false
  local normalize = opts.normalize or false

  local vector = torch.FloatTensor(300):zero()
  local num_words = 0
  local eps = 1e-9
  for word in sentence:gmatch('%w+') do 
    if w2v.words[word] == nil then
      vector:add(w2v.vec[w2v.words['UNK']])
    else
      vector:add(w2v.vec[w2v.words[word]])
    end
    num_words = num_words + 1
  end

  if average then
    vector:div(num_words + eps)  
  end

  if normalize then
    vector:div(vector:norm() + eps)
  end

  return vector
end


return embedding_utils