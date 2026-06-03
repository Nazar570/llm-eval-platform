# Label Studio annotation setup

Label Studio provides the human-in-the-loop labels that `training.dataset_builder`
combines with weak supervision to build the judge training set.

## 1. Start the stack

```bash
make up
```

Label Studio is available at http://localhost:8080 (admin@eval.local / labelstudio).
A static API token (`label-studio-local-token`) is provisioned through the
`LABEL_STUDIO_USER_TOKEN` environment variable so scripts can authenticate without
the UI.

## 2. Create the annotation project

Create a project titled `llm-eval-annotation` with the following labelling
interface. The choice values must match `training.positive_choice_labels`
(`good`, `correct`, `pass`) so they map to the positive class.

```xml
<View>
  <Text name="question" value="$question"/>
  <Text name="answer" value="$answer"/>
  <Choices name="quality" toName="answer" choice="single">
    <Choice value="good"/>
    <Choice value="bad"/>
  </Choices>
</View>
```

## 3. Import tasks

`sample_tasks.json` in this directory contains question/answer pairs ready to
import from the UI (Import > Upload Files) or via the API:

```bash
curl -s -X POST "http://localhost:8080/api/projects/1/import" \
  -H "Authorization: Token label-studio-local-token" \
  -H "Content-Type: application/json" \
  --data @services/annotation/sample_tasks.json
```

## 4. Rebuild the dataset

After annotating, pull the labels and regenerate the training set:

```bash
PYTHONPATH=. python -m training.dataset_builder
```

The builder exports annotations from project `LABEL_STUDIO_PROJECT_ID`, merges
them with weak labels derived from stored evaluation scores, deduplicates, and
writes `data/eval_dataset.jsonl`.
