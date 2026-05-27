SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions based ONLY on the provided context documents.

Rules:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. Cite your sources using bracketed references like [1], [2], etc.
3. Each factual claim must have at least one citation.
4. If the context does not contain enough information to answer, say so explicitly.
5. Be concise and direct."""

CONTEXT_TEMPLATE = """Context documents:

{context_blocks}

Question: {question}

Answer (cite sources using [1], [2], etc.):"""

CITATION_VERIFY_PROMPT = """Determine if the source text supports the claim. Respond with only "SUPPORTED" or "NOT_SUPPORTED".

Claim: {claim}

Source text: {source}

Verdict:"""

CONFIDENCE_PROMPT = """Rate how completely this answer addresses the question on a scale of 0-10.
Only respond with a single integer number.

Question: {question}

Answer: {answer}

Completeness score (0-10):"""
