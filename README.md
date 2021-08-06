

## MLP XOR



<img src="images/xor_output.png" width=300/>

최종 결과 O를 출력했을 때 결과가 [ 0, 1, 1, 0 ]에 근접하기 때문에 XOR 연산을 만족하는 것을 알 수 있다.

loss trace 그래프는 다음과 같다. 

<img src="images/xor_loss.png" width=500/>

<br>

<img src="images/final_loss.png" width=700/>

interation 횟수는 4000, learning rate는 0.1, 같은 조건으로 학습을 해도 최종 loss는 조금씩 다르다. 랜덤한 숫자로 initialize 하기 때문에 학습이 반복되면서 최종 결과 값은 비슷해 지지만 완전히 같은 값이 나오지는 않는다.

![img](file:///C:/Users/chsj/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)train 함수에 왼쪽 코드를 추가해서 loss가 0.02 미만이 되는 지점의 iteration 횟수를 tmp에 담아 리턴했다. 

![img](file:///C:/Users/chsj/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)learning rate = 0.5일 때 결과다. iteration 854번째에 loss가 0.02 미만이 되었다.

![img](file:///C:/Users/chsj/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)learning rate = 0.1일 때 결과다. iteration 3268번째에 loss가 0.02 미만이 되었다.

![img](file:///C:/Users/chsj/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)learning rate = 0.01일 때 결과다. iteration 32391번째에 loss가 0.02 미만이 되었다.

Learning rate에 따라 학습되는 속도가 달라지기 때문에 learning rate가 클수록 적은 iteration 횟수로도 loss가 0.02 미만에 도달하는 것을 확인할 수 있다.



Weight와 bias를 모두 0으로 initialize 되도록 코드를 수정하고 결과를 다시 출력했다.

Loss trace, 최종 loss, 결과인 O를 순서대로 출력했다.

<img src="images/loss_trace.png" width=500/>