from ReadingCon import util, predicting_dev, args

def DataProcess(text,question):
    util.creat_test_example(text, question,
                            result=args.predict_example_files)
    best_answer = predicting_dev.eval_all()
    if best_answer == "":
        best_answer = "无"
    return best_answer

#test
if __name__ == '__main__':
    DataProcess('妈妈在家','谁在家？')