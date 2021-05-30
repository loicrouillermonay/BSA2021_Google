## **Home**

Returns "BSA2021 - Team Google's API is working!" if the API is working.

- **URL**

  /

- **Method:**

  `GET`

- **URL Params**

  None

- **Data Params**

  None

- **Success Response:**

  - **Code:** 200 <br />
    **Content:** `BSA2021 - Team Google's API is working!`

## **Sentence prediction**

Returns the predicted difficulty of the text input and the original text input.

- **URL**

  /api/predict

- **Method:**

  `GET`

- **URL Params**

  **Required:**

  `text=[string]`

- **Data Params**

  None

- **Success Response:**

  - **Code:** 200 <br />
    **Content:** `{"difficulty":"CEFR_LEVEL","text":"INPUT"}` <br />
    with _CEFR_LEVEL_ the level of difficulty of the _INPUT_ according to the [Council of Europe](https://www.coe.int/en/web/common-european-framework-reference-languages/table-1-cefr-3.3-common-reference-levels-global-scale])

## **Words prediction**

Returns the predicted difficulty of each word of a text input and the original text input.

- **URL**

  /api/predict/words

- **Method:**

  `GET`

- **URL Params**

  **Required:**

  `text=[string]`

- **Data Params**

  None

- **Success Response:**

  - **Code:** 200 <br />
    **Content:** `{"difficulty":"[CEFR_LEVELS]","text":"INPUT"}` <br />
    with _CEFR_LEVELS_ the level of difficulty of each word from _INPUT_ according to the [Council of Europe](https://www.coe.int/en/web/common-european-framework-reference-languages/table-1-cefr-3.3-common-reference-levels-global-scale])
