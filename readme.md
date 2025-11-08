# Face Verification API

POST `/compare-faces`

Form data:
- image1: first face image
- image2: second face image

Response:
```json
{ "match": true, "distance": 0.23 }
