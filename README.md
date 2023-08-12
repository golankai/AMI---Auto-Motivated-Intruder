# Auto-Intruder
Automatic Re-identifying tool

## Tasks

### Organization

##### De-Anonymizer
1. Define 3-5 process with different characteristics.
2. Eval and analyze the results on adele
3. Select the top 2-3 and execute on all dataset.
4. Eval and analyze the results.

##### Scores
1. Try few shots.
2. eval and compare to roberata fine-tuning results.
3. maybe we want to think on another way. 
    we can extract characteristics that makes the prediction harder for the llm (i.e. number of persons in the text)
    we can think about a (manual) methodology for give approx. score (i.e. number of unique chars, etc.)


##### Report
1. introduction
2. related works
3. data description
4. solution - general approach (maybe ?)



### Code
1. Intruder - prompting about the anonymized text
1.1 Scoring and identifying key factors

2. Check with other models

3. Enssemble -> Score
