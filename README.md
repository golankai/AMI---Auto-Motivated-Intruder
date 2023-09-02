# Auto-Intruder
Automatic Re-identifying tool


The burgeoning demand for expansive text data in social and medical research presents a challenge: while pressing questions necessitate using sensitive data, data protection and ethics raise a barrier.
A promising solution lies in automated text anonymization, which performs data masking of Personally Identifiable Information ("**PII**"), such as names and locations, at scale.
This work displays experiments on the topic, utilizing LLMs capabilities, as an off-the-shelf tool.

Until today, numerous studies illustrated that text can be anonymized with only minor performance degradation.
Nevertheless, this approach is not infallible, especially for sensitive sectors such as healthcare.
### How can we tell if our text is sufficiently anonymized?
The domain of text anonymization lacks a standardized evaluation method, rendering the challenge multifaceted and open to individual interpretation.
Traditionally, the most reliable evaluators have been humans attempting to re-identify individuals from anonymized texts, this process is called the \underline{Motivated Intruder Test}.
There are also some more technical metrics, such as the accuracy of tokens to be redacted, but again, this is prone to biases and personal interpretation.

Our aspiration is to surpass human re-identification capabilities, thereby setting a new benchmark for evaluating anonymized data.
Namely, our approach can be used as an evaluation technique for other anonymization schemes, manual and automatic.

Our motivation is that a spontaneous re-identification process done by humans is indeed a threat, but in the world of Big Data, is not efficient and scalable.
An automated version is very likely what will be developed and used by a potential Adversary.

### Problem Statement:
- RQ1. Compete with humans in the task of re-identifying famous persons from anonymized texts.
- RQ2. Develop a new metric of the anonymization (/re-identification) rate of a single text.

We chose this problem as data collection and data protection are two very relevant and essential topics, and their convergence is not yet close to being fully discovered.
