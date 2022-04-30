//黑客弹幕模块
//控件重新赋值模块
//声明全局数组，用于存放取值


 window.onload = function(){

     //获取画布对象
                var canvas = document.getElementById("canvas");
                //获取画布的上下文
                var context =canvas.getContext("2d");
                //获取浏览器屏幕的宽度和高度
                var W = window.innerWidth;
                var H = window.innerHeight;
                //设置canvas的宽度和高度
                canvas.width = W;
                canvas.height = H;
                //每个文字的字体大小
                var fontSize = 16;
                //计算列
                var colunms = Math.floor(W /fontSize);
                //记录每列文字的y轴坐标
                var drops = [];
                //给每一个文字初始化一个起始点的位置
                for(var i=0;i<colunms;i++){
                    drops.push(0);
                }

                //运动的文字
                var str ="javascript html5 canvas";
                //4:fillText(str,x,y);原理就是去更改y的坐标位置
                //绘画的函数
                function draw(){
                    context.fillStyle = "rgba(0,0,0,0.05)";
                    context.fillRect(0,0,W,H);
                    //给字体设置样式
                    context.font = "700 "+fontSize+"px  微软雅黑";
                    //给字体添加颜色
                    context.fillStyle ="#00cc33";//可以rgb,hsl, 标准色，十六进制颜色
                    //写入画布中
                    for(var i=0;i<colunms;i++){
                        var index = Math.floor(Math.random() * str.length);
                        var x = i*fontSize;
                        var y = drops[i] *fontSize;
                        context.fillText(str[index],x,y);
                        //如果要改变时间，肯定就是改变每次他的起点
                        if(y >= canvas.height && Math.random() > 0.99){
                            drops[i] = 0;
                        }
                        drops[i]++;
                    }
                }

                function randColor(){
                    var r = Math.floor(Math.random() * 256);
                    var g = Math.floor(Math.random() * 256);
                    var b = Math.floor(Math.random() * 256);
                    return "rgb("+r+","+g+","+b+")";
                }

                draw();
                setInterval(draw,30);
	 			var container = document.getElementById('container');
            var list = document.getElementById('list');
            var buttons = document.getElementById('buttons').getElementsByTagName('span');
            var prev = document.getElementById('prev');
            var next = document.getElementById('next');
            var index = 1;
            var timer;
			
            function animate(offset) {
                //获取的是style.left，是相对左边获取距离，所以第一张图后style.left都为负值，
                //且style.left获取的是字符串，需要用parseInt()取整转化为数字。
                var newLeft = parseInt(list.style.left) + offset;
                list.style.left = newLeft + 'px';
                //无限滚动判断
                if (newLeft > -820) {
                    list.style.left = -4100 + 'px';
                }
                if (newLeft < -4100) {
                    list.style.left = -820 + 'px';
                }
            }

            function play() {
                //重复执行的定时器
                timer = setInterval(function () {
                    next.onclick();
                }, 2000)
            }

            function stop() {
                clearInterval(timer);
            }

            function buttonsShow() {
                //将之前的小圆点的样式清除
                for (var i = 0; i < buttons.length; i++) {
                    if (buttons[i].className == "on") {
                        buttons[i].className = "";
                    }
                }
                //数组从0开始，故index需要-1
                buttons[index - 1].className = "on";
            }

            prev.onclick = function () {
                index -= 1;
                if (index < 1) {
                    index = 5
                }
                buttonsShow();
                animate(820);
            };

            next.onclick = function () {
                //由于上边定时器的作用，index会一直递增下去，我们只有5个小圆点，所以需要做出判断
                index += 1;
                if (index > 5) {
                    index = 1
                }
                animate(-820);
                buttonsShow();
            };

            for (var i = 0; i < buttons.length; i++) {
                (function (i) {
                    buttons[i].onmousemove = function () {

                        /*  这里获得鼠标移动到小圆点的位置，用this把index绑定到对象buttons[i]上，去谷歌this的用法  */
                        /*  由于这里的index是自定义属性，需要用到getAttribute()这个DOM2级方法，去获取自定义index的属性*/
                        var clickIndex = parseInt(this.getAttribute('index'));
                        var offset = 820 * (index - clickIndex); //这个index是当前图片停留时的index
                        animate(offset);
                        index = clickIndex; //存放鼠标点击后的位置，用于小圆点的正常显示
                        buttonsShow();
                    }
                })(i)
            }

            container.onmouseover = stop;
            container.onmouseout = play;
            play();



            };




