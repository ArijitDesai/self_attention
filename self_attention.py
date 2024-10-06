from gensim.models import Word2Vec
import math
import torch

# Let's say we have a simple sentence: "I am new to Deep Learning"
# We want to see how each word in this sentence "talks" or "pays attention" to other words.

text = "I am new to Deep Learning"
sentence_list = text.split(" ")  # Split the sentence into individual words (tokens).
sentence = [sentence_list]  # Create a list of lists to match what Word2Vec expects.

# We want to represent each word with numbers, called "word embeddings". 
# Think of each word having its own special set of numbers.
# So, we use Word2Vec to generate these special numbers (word embeddings) for each word in our sentence.

# We tell Word2Vec to create 5 special numbers (embeddings) for each word.
word_embeddings = []
model = Word2Vec(sentences=sentence, vector_size=5, window=3, min_count=1, sg=1, workers=1)

# Let's take these special numbers (embeddings) for each word and store them in a list.
for word in sentence[0]:
  word_embeddings.append(model.wv[word])

# Convert the list of word embeddings into something PyTorch can understand, called a tensor.
# Think of this like transforming our list of numbers into a format that PyTorch can work with.
word_embedding_tensor = torch.tensor(word_embeddings, dtype=torch.float32)


# Now we need to define some weights. These weights are like special "glasses" that help us look at our word embeddings
# differently. We have three sets of glasses: one for Queries, one for Keys, and one for Values.
# Each weight matrix will be 5x5 because we want to keep the size of our "view" (5) the same.

# Let's think of Query, Key, and Value in terms of a **library database analogy**:
# 1. **Query**: A query is like a **search request** you send to the library. For example, "Show me all books written by Author X".
# 2. **Key**: A key is like the **index or catalog** in the library. It helps you match your query against different books.
# 3. **Value**: A value is the **actual book information** you get as a response. It contains the data you're looking for, like the book's title, year, etc.

# In our case, each word in the sentence will have its own Query, Key, and Value.
# We will use the Query to compare with the Keys of other words to determine how much attention it should pay.
# The Value will be used to collect the actual information based on the attention weights.

torch.manual_seed(42)  # Use this to make sure our random weights are always the same.
W_Q = torch.randn(5, 5)  # Randomly create a weight matrix for Query. This will be like our "search request generator".
W_K = torch.randn(5, 5)  # Randomly create a weight matrix for Key. This is like our "library catalog index".
W_V = torch.randn(5, 5)  # Randomly create a weight matrix for Value. This is like our "book information".

# We'll use the square root of the size of our embeddings (5) to make our calculations more stable.
# This number will be used later when calculating attention scores.
scaling_factor = math.sqrt(5)

# Let's put on our "glasses" and look at our word embeddings differently!
# Each word embedding gets multiplied by each set of weights to get new versions: Query, Key, and Value vectors.
# - Query: Like a search query you send to the library to find related books.
# - Key: Like the library catalog, helping you match your query with different books.
# - Value: Like the actual book data you get when you find a match in the catalog.

word_embedding_query = torch.matmul(word_embedding_tensor, W_Q)  # Shape: (6, 5) - 6 words, each with a 5D query vector.
word_embedding_key = torch.matmul(word_embedding_tensor, W_K)    # Shape: (6, 5) - 6 words, each with a 5D key vector.
word_embedding_value = torch.matmul(word_embedding_tensor, W_V)  # Shape: (6, 5) - 6 words, each with a 5D value vector.


# Now, let's make our words "talk" to each other using Queries and Keys.
# Imagine each word sends out a search request (Query) to see which books (Keys) match its search criteria.
# The more a Query matches a Key, the more attention that word pays to the other word.

# We'll create a list to store what each word "understands" after paying attention to the other words.
context_vector_list = []

# Go through each word's query (like each word sending its own search request).
for query in word_embedding_query:
    # This list will store how much the current word's query matches the keys of other words.
    word_similarity_scaled_score_list = []
    
    # Compare this word's query to the key of each word in the sentence.
    for tensor in word_embedding_key:
        # Compute dot product between the current query and a key vector.
        # This is like checking how closely your search request (Query) matches a book in the library (Key).
        word_similarity_score = torch.matmul(query, tensor)
        
        # Scale down the similarity score to prevent too big numbers (this makes sure the scores are balanced).
        word_similarity_scaled_score_list.append(word_similarity_score / scaling_factor)
    
    # Now we have similarity scores between this word and all other words.
    # But these scores are not in a nice 0 to 1 range yet, so we use softmax to convert them into "attention weights".
    softmax = torch.nn.Softmax(dim=0)  # Softmax turns the scores into values between 0 and 1 that add up to 1.
    attention_weights = softmax(torch.tensor(word_similarity_scaled_score_list, dtype=torch.float32))  # Shape: (6,)

    # The attention weights tell us how much attention the word pays to every other word.
    # Now, we take each attention weight and multiply it with the corresponding value vector to get the context vector.
    # This is like taking the amount of attention and using it to decide how much information to take from each book.
    context_vector_list.append(torch.matmul(attention_weights, word_embedding_value))  # Shape: (5,)

# Let's see the new context vectors for each word.
# These context vectors tell us what each word "thinks" after considering all other words in the sentence.
for i, context_vector in enumerate(context_vector_list):
    print(f"Context vector for word '{sentence[0][i]}': {context_vector}")
