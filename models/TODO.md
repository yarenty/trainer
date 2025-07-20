
# Important parts

## Modelfile Update

Modelfile should now includes:
```
PARAMETER stop "### End"
```
This tells Ollama to stop generating when it sees the end marker, preventing runaway output.
