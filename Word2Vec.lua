local wordemb = require 'wordemb'


local Word2Vec = torch.class('Word2Vec')


function Word2Vec:__init(filepath)
  self.w2v = wordemb.load_word2vec_bin(filepath)
end


function Word2Vec:encode(sentence, opts)
  opts = opts or {}
  local average = opts.average or false
  local normalize = opts.normalize or false

  local vector = torch.FloatTensor(300):zero()
  local num_words = 0
  local eps = 1e-9
  for word in sentence:gmatch('%w+') do 
    if self.w2v.words[word] == nil then
      vector:add(self.w2v.vec[self.w2v.words['UNK']])
    else
      vector:add(self.w2v.vec[self.w2v.words[word]])
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
