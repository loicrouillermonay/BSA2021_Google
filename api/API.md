**Home**
----
  Returns "BSA2021 - Team Google's API is working!" if the API is working.

* **URL**

  /

* **Method:**

  `GET`

* **URL Params**

  None

* **Data Params**

  None

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `BSA2021 - Team Google's API is working!`



**Prediction**
----
  Returns the overall text input prediction and the text input.

* **URL**

  /api/predict

* **Method:**

  `GET`

* **URL Params**

  **Required:**
 
   `text=[string]`

* **Data Params**

  None

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `{"difficulty":"CEFR_LEVEL","text":"INPUT"}` 
    with _CEFR_LEVEL_ the level of difficulty according to the (Council of Europe)[https://www.coe.int/en/web/common-european-framework-reference-languages/table-1-cefr-3.3-common-reference-levels-global-scale]
