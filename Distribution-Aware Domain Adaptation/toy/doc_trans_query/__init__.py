a = 1
def f():
    print(a)

"""
Primary lightbulb of doc_trans_query is

# Current method is bad...
    QGen for DA (Domain Adaptation) in IR (Information Retrieval) is bad.

# Then how and what is bad?
1. QGen models are trained on ID (In Domain) data.
    Therefore, their generation does not reflect the OOD corpus.
    Such as, the vocabulary distribution of OOD corpus.

2. QGen models do not know what the task is
    Therefore, their generated queries are not like the gold query of the OOD dataset.
    This can be overcome by utilizing the LLM that can do in-context learning via the given prompt.

3. QGen models only watch the document in their input length.
    This is something obvious to all BERT based models.
    Though, utilizing the LLM model requires huge computation.

# What do we want to solve?
The first one.
We can improve first one easily than others as they require to use LLM.

# How to solve?
1. Inject the Vocab distribution to the QGen model.
    However, we have seen that this is quite difficult to do in a sense of we have done in DADA.
    Therefore, we have to figure out some way to do this like RamDA.
        But, we also have slightly viewed that RamDA is not that good.

2. Exclude the QGen model and mask some terms on the given document.
First, we know that QGen's objective is to generate the query that make the given document to be the relevant.
Therefore, some researches either enforce that the given document is top1 (GPL)
    or filter (Promptagator) the generated query if it is not making the given document to be the top1.
Furthermore, the GPL has to label the document within the contrastive learning framework.

These approaches are too expensive.
And the QGen is a detour to generate a query that is relevant to the document.

We can make the relevant query in more cheap way.
Just take the sentence from the document and mask some random terms.

This approach requires some justification.
    1. The query distribution is similar to the relevant document distribution but not the same.
        # proof:
        # Sim: 5.6670597034513825e-12
        # Not Sim: 1.7789145011334943e-09
        
        The query is the surface expression of the query intent.
        Therefore, the query will be more either vague or general than the document.
        Plus, the query is in most case shorter than the document.
        In a nutshell, the query distribution might be the subset of the relevant document distribution.
    
    2. The attention between query and document when it is match is not similar to the lexical matching.
        # proof:
        # Sim: 0.10619469026548672
        # Not Sim: 0.10793898957618543
        
        Although we have some related works (***) that shows the dense model can capture semantic matching more.
        # But still, in some case the lexical matching is more interpretable and effective. (SPLADE)
        # Therefore, we can utilize the lexical matching to generate the query that is similar to the document.
        
        At this point, we need to check do the dense models can capture the lexical matching.
        If the dense models capture some lexical matching, then we cannot mask any random terms.
        Of course, there will be some pattern in semantic matching but it should be more vague than lexical matching.
        Therefore, we can mask some random terms.

# Setup
BeIR

# Method
1. Obtain the document's relevant documents.
    We can use the TF-IDF/BM25 to obtain the relevant documents.
    TF-IDF and BM25 is the cheapest way to get the relevant document list.
    (The model can be converted to other NN models but lets use the cheaper one.)
    (Furthermore, unlike other NN models, TF-IDF/BM25 can utilize all the sentences from the document.)
    
    return {
        doc1_id: {
            doc2_id: score,
            doc3_id: score,
            ...
        }
    }

2. Now we slice the document into sentences.
    The doc1 is sliced into sentences.
    Here is the concern, which sentence should be treated as the query?
    We can use multiple ways:
        1. Randomly select the sentence.
        2. Select all sentences.
            This can be a problem if the document is too long.
        3. Select the sentence that has the highest score.
            Take all the sentences and then select the sentence based on the retrieval result.
            If the sentence that retrieves the given document (doc1) is the top1, then we can select that sentence.
                (this lightbulb is similar to promptagator)
            This approach might be plausible as it can have the score at the same time.

    return {
        doc1_id: {
            doc1_q1_id: query1,
            doc1_q2_id: query2,
            ...
        }
    }

(3. Now label)
    return {
        doc1_id: {
    }


"""