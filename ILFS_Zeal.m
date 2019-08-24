function [Rank, Score] = ILFS_Zeal(train_X, train_y, TT, verbose)


    if (nargin < 3)
        verbose = 0;
        TT = 3;
    end
    if (nargin < 4)
        verbose = 0;
    end
    
    num_F = size(train_X, 2);
    
    A = LearningGraphWeights(train_X, train_y, TT, verbose);
    
    priori_len = ceil(max(A * ones(length(A), 1))) / num_F;
    factor = 0.99;
    
    %% Letting paths tend to infinite: Inf-FS Core
    if (verbose)
        fprintf('Letting paths tend to infinite \n');
    end
    
    I = eye(size(A, 1));  % Indentity Matrix
    rho = max(eig(A));
    r = factor / rho;  % Set a meaningful value for r
    y = I - (r * A);
    
    S = inv(y) - I;  % see Gelfand's formula - convergence of the geometric series of matrices
 
    %% Estimating energy scores
    if (verbose)
        fprintf('Estimating relevancy scores \n');
    end
    Score = sum(S, 2);  % prob. scores s(i)
    
    %% Ranking features according to s
    if (verbose)
        fprintf('Features ranking \n');
    end
    
    [~, Rank] = sort(Score, 'descend');
    Rank = Rank';
    Score = Score';

end