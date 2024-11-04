### Running 

for now it is 

```
python3 watershed.py ./hall2.jpg 4 3 1 5 2
```

### Future work

Replaced thresholding with canny - at least now im able to segment smth.
Need to play with first dilude-erode on canny, and on second dilude.

Maybe, move them into args instead of now-useless denoise params
