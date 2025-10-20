Understand and implement landscapes for:
 - [] understand landscape classifications
 - [] generate landscape with parameters of table 9.1 (p. 140)

implement performance profile ($P_{RED}$):

implement cooperative co-evolutionary algorithms:
 - CCHyperMGA
 - CCRIGA

compare the following algorithms:
 - CCHyperM
 - CCRIGA
 - HyperM
 - RIGA

| Parameter             | Static   | Progressive | Abrupt | Chaotic |
| --------------------- | -------- | -------- | -------- | -------- |
| Domain                | [0, 100] | [0, 100] | [0, 100] | [0, 100] |
| Number of dimensions  | 5        | 5        | 5        | 5        |
| Peak count            | 10       | 10       | 10       | 10       |
| Peak height           | [30, 70] | [30, 70] | [30, 70] | [30, 70] |
| Peak width            | [1, 12]  | [1, 12]  | [1, 12]  | [1, 12]  |
| Height change severity | 1       | 1        | 10       | 10       |
| Width change severity | 0.05     | 0.05     | 0.05     | 0.05     |
| Change severity       | 1        | 1        | 10       | 10       |
| Random movement % (𝜆) | 0        | 0        | 0        | 0        |
| Change frequency (iterations) | ∞ | 20      | 100      | 30       |


Domain               -  > domain: Tuple[float, float] = (0.0, 100.0),
Number of dimensions -  > dimension: int = 2,
Peak count           -  > num_peaks: int = 10,
Peak height          -  > min_height: float = 30.0, max_height: float = 70.0,
Peak width           -  > min_width: float = 1.0, max_width: float = 12.0,
Height change severity -> height_severity: float = 7.0,
Width change severity  -> width_severity: float = 1.0,
Change severity        -> change_severity: float = 1.0,
Random movement % (𝜆)  -> lambda_param: float = 0.0,
Change frequency       -> change_frequency: int = 5000,
