HOMOPHONE_MAPPING_PROMPT = """\
For each label below, list 3-5 English homophones (words that sound like the label).
Return ONLY a valid JSON object: keys are labels, values are arrays of strings.
Labels: {labels}"""

SYNONYM_MAPPING_PROMPT = """\
For each label below, list 3-5 synonyms or closely related object names a vision model might use.
Return ONLY a valid JSON object: keys are labels, values are arrays of strings.
Labels: {labels}"""
