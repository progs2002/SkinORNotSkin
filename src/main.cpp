#include "train.h"
#include "predict.h"

int main(int argc, char const *argv[])
{
    if(argc > 2)
    {
        if(strcmp(argv[1],"train") == 0)
        {    
            //train the model
            if(train_init(argv[2]) == 1)
                LOG("cannot initialize trainer");
            else
                LOG("trainer initialized");
        }
        else
            LOG("invalid argument");
    }
    else
        //TODO: add code to check if model is trained or not
        predict();   
    return 0;
}