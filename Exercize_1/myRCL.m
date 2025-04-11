function dx = myRCL(~, x)
    global R C L V

    iL = x(1);
    vC = x(2);
    
    % System equation
    diL = (1/L)*(V-R*iL-vC);
    dvC = (1/C)*iL;
    dx = [diL; dvC];
end