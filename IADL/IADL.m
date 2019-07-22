%--------------------------------------------------------------------------
%|                        __   __   ____  __                              |
%|                       (  ) / _\ (    \(  )                             |
%|                        )( /    \ ) D (/ (_/\                           |
%|                       (__)\_/\_/(____/\____/                           |
%|                                                                        |
%+------------------------------------------------------------------------+
%|               Information Assisted Dictionary Learning                 |
%+------------------------------------------------------------------------+
%|   IADL is designed to solve the proposed constrained optimization task |
%| via the Majorized Minimization rationale. This algorithm imposes two   |
%| major kind of contraints: the similarity constraint over the dictionary|
%| and a row-wise sparsity constraint via the weighted l1-norm.           |
%+------------------------------------------------------------------------+
%| Manuel Morante Moreno                                                  |
%|                                              Last Update: 22 Jul  2019 |                                           
%+------------------------------------------------------------------------+
%|                                -mMm-                                   |
%+------------------------------------------------------------------------+
%|                                                                        |
%|   === MAIN PARAMETERS ==============================================   |
%|   param.data   -> double T x N matrix with the data                    |
%|   param.K      -> Number of components                                 |
%|   param.iter   -> Number of iterations                                 |
%|   praam.Lam    -> double K vector with the percent of sparsity of each |
%|        spatial map with values in [0,100]                              |
%|        NOTE: 0% Means that the spatial map is dense (no zeros)         |
%|                                                                        |
%|   === OPTIONAL PARAMETERS ==========================================   |
%|   param.Verb -> Display the process (y/n)                (default  'y')|
%|   param.Ini  -> Set the initialization method                (def. rnd)|
%|                                                                        |
%|   param.Prep -> Apply PCA for dimensionality reduction (y/n)   (def. y)|
%|   param.pRed -> double ?? [0,1]. Percent of the reduction      (def. 0)|
%|                                                                        |
%|   --- Split Control ---                                                |
%|   param.merge -> Merge splits during the initialization (y/n)  (def. n)|
%|                                                                        |
%|   --- Canonical Dictionary ---                                         |
%|   param.Del  -> double T x M matrix which contains the canonical atoms |
%|        for the constrained par of the dictionary                       |
%|   param.cdl  -> parameter of the proximity of the canonical atom with  |
%|        respect to the current estimated atom                           |
%|                                                                        |
%|   --- Spatial Maps ---                                                 |
%|   param.Ts -> Number of internal spatial iterations            (def. 1)|
%|   --- Split Control ---                                                |
%|   param.SpC -> Check for split after the decomposition (y/n)   (def. n)|
%|                                                                        |
%|   --- Dictionary ---                                                   |
%|   param.ccl   -> Value of the normalization of each atom       (def. 1)|
%|   param.Td    -> Number of internal temporal iterations        (def. 1)|
%|                                                                        |
%+------------------------------------------------------------------------+
%|   === RETURN =======================================================   |
%|   D -> double T x K matrix with the Dictionary                         |
%|   S -> double K x N matrix with the Coefficient matrix                 |
%|                                                                        |
%|   --- Optional ---                                                     |
%|   E -> A vector with the loss error per iteration (time consuming)     |
%--------------------------------------------------------------------------
function [D,S,E] = MoM_New(param) % Main Function

% === PARAMETER SET =======================================================

	% Check Verbose
	if(~isfield(param,'Verb'))
		Verb = true;
	else
		if(strcmp(param.Verb,'y'))
			Verb = true;
		else
			Verb = false;
		end
	end

	if(Verb) 
		fprintf('    ___   _   ___  _\n   |_ _| /_\ |   \| |\n');   
 		fprintf('    | | / _ \| |) | |__   |___/_/ \_\___/|____|\n');
		fprintf('_____________________________________\n\n');
		fprintf('   - Parameters ------------ [  ]');
	end

	%%% Basic parameters %%%
	[T,~] = size(param.data);    % Number of time components and voxels
	K     = param.K;             % Number of sources
	Tt    = param.iter;          % Number of iterations

	% Check Initialization mode
	if(~isfield(param,'Ini'))
		param.Ini = 'rnd';
	end

	% Merge Splits
	if(~isfield(param,'merge'))
		param.merge = 'n';
	end

	% Check normalization value
	if(~isfield(param,'ccl'))
		param.ccl = 1;
	end

	% Check Intertal loops
	if(~isfield(param,'Ts'))
		param.Ts = 1;
	end
	if(~isfield(param,'Td'))
		param.Td = 1;
	end

	% Check Dimensionality Reduction
	if(~isfield(param,'Prep'))
		param.Prep = 'n';
	else
		if(~isfield(param,'pRed'))
			param.pRed = 0;
		end
	end

	% Check Error
    if(nargout==3)
		E = zeros(Tt,1);
    end

    if(Verb) 
        fprintf('\b\b\bOk]\n');
        
        % Steps for the verbose stuff
		Vstp = floor(linspace(1,Tt,20));
    	vi = 1;
    end

	% --- Dimensionality Reduction -----------------------------------------
	if(param.Prep=='y')

		if(Verb); fprintf('   - Dim. Reduction --------- [  ]'); end

		% Initializations
		T = max(floor(T*param.pRed),K);

		% Compute the decomposition in the reduced space
		[U,~,~] = svds(param.data*param.data',T);

		% New dataset
		Y = U'*param.data;
		param.data = Y;

		clear('Y');

		% Readjust the size of the canonical Dictionary
		if(isfield(param,'Del'))
			param.Del = U'*param.Del;
		end

		if(Verb); fprintf('\b\b\bOk]\n'); end
	end

% === INITIALIZATION ======================================================
    
    if(Verb); fprintf('   - Initialization -------- [  ]'); end

    % Call Initialization Function
    [D,S] = InitiaMe(param,param.Ini);

    if(Verb); fprintf('\b\b\bOk]\n'); end

% === MERGE SPLITS ========================================================
	
	if(param.merge=='y')
		if(Verb); fprintf('   - Splits: [...]'); end

		[D,S,Nmrg] = MergeMe(D,S,1);

		if(Verb); fprintf('\b\b\b\b%3i] --------- [Ok] \n',Nmrg);end
	end

% === CANONICAL DICTIONARY (Safe mode) ====================================
	if(isfield(param,'Del'))

		if(Verb); fprintf('   - Setting Can. Dict. ---- [  ]'); end

		% Parameters
		Del   = param.Del;   % Canonical Dictionary
		[~,M] = size(Del);   % Number of assisted atoms

		Daux = [Del D];

		Crr = corrcoef(Daux);

		% Find the most correlated atoms and permute them
		for i = 1:M
			[~,j] = max(abs(Crr(i,M+1:end)));

			% Permute and impose
			D(:,j) = D(:,i);
			D(:,i) = Del(:,i);

			Saux = S(j,:);
			S(j,:) = S(i,:);
			S(i,:) = Saux;

			% Remove the permuted column
			Crr(:,j+M) = zeros(M+K,1);
		end

		% Check the cdl parameters
		if(length(param.cdl)<M)
			param.cdl = param.cdl*ones(1,M);
		end

		if(Verb); fprintf('\b\b\bOk]\n'); end

		clear('Ord');
	end

% ======== ORDERING =======================================================

	if(Verb); fprintf('   - Ordering -------------- [      ]'); end
	
	% Initialization
	Elim = 0.005;     % Limit error
	Es   = 1;         % Assumed Initial error
	cnt  = 1;         % Iteration counter
	Lit  = 666;       % Limit of iterations

	Ord  = param;     % Set parameter for Ordering


	if(isfield(param,'Del'))
		[~,M] = size(param.Del);

		Ord.cdl = zeros(1,M); % Fix canonical atoms
		
		M = M + 1;    % Avoid ordering canonical atoms
	else
		M = 1;
	end

	% Ordering Loop
	while(Es>=Elim)&&(cnt<=Lit)

		Sn = NewCoefUni(S,D,Ord);
		D  = NewDict(Sn,D,Ord);

		Es = norm(S-Sn)/norm(S);
		S = Sn;

		cnt = cnt + 1;

		if(Verb);Err = floor(100*(log(Es)/log(Elim)));end
		if(Verb);fprintf('\b\b\b\b\b\b%3i %%]',max(0,Err));end
	end

	clear('Ord');

	% Order all the sources according to their relative energy
	[D(:,M:K),S(M:K,:)] = ReorderMe(D(:,M:K),S(M:K,:),param.Lam(M:K));


	if(Verb);fprintf('\b\b\b\b\b\b\bOk]\n'); end


% === MAIN LOOP ===========================================================
	
	if(Verb);fprintf('   Progress: \n'); end
	if(Verb);fprintf('     >> [                    ] 00.0 %%');end

    for t = 1:Tt

		S = NewCoef(S,D,param);   % Update Coefficient

		D = NewDict(S,D,param);   % Update Dictionary

		% Error
		if(nargout==3)
			E(t) = sqrt(norm(param.data-D*S,'fro'));
		end

		% --- Verbose Stuff -----------------------------------------------
        if (Verb)
            if(t == Vstp(vi))
        		vprct = t*100./Tt;
        		for i=1:28; fprintf('\b'); end
        		for i=1:vi; fprintf('=');  end
        		for i=(vi+1):20; fprintf(' ');  end
        		fprintf('] %4.1f %%',vprct);
        		vi = vi+1; 
            else
                if(mod(t,3)==0 || t==Tt)
        		vprct = t*100./Tt;
        		fprintf('\b\b\b\b\b\b%4.1f %%',vprct);
                end
            end
        end
        %------------------------------------------------------------------
    end

% === POST-PROCESSING =====================================================

	if(Verb);fprintf('\n'); end

	% --- Recover Dimensionality Reduction ---
	if(param.Prep=='y')

		D = pinv(U')*D;

		if(Verb); fprintf('   - Recover Dimensions ------- [Ok]\n'); end
	end

	%--- Merge potential final Splits ---
	if(isfield(param,'SpC'))
		if(param.SpC=='y')
			if(Verb); fprintf('   - Splits: [...]'); end

			[D,S,Nmrg] = MergeMe(D,S,1);

			if(Verb); fprintf('\b\b\b\b%3i] --------- [Ok] \n',Nmrg);end
		end
	end

	if(Verb); fprintf('\n   - Completed!  \\(^ ^ )\n\n'); end

end

%==========================================================================
%||                     INTERNAL FUNCTIONS                               ||
%==========================================================================

%--------------------------------------------------------------------------
%|      function NewCoef (So,D,param)                                     |
%|    Calculates the new coefficients imposing sparsity per rows using    |
%| the weighted l1-norm projection.                                       |
%--------------------------------------------------------------------------
function [S] = NewCoef(So,D,param)

	% Main parameters
	Ts = param.Ts;     % Internal Loops
	[K,N] = size(So);  % Dimensions

	epsilon = 1e-6;    % Stability control

	%----------------------------------------------------------------------
	%    NOTE: Lam is given by the expected percent of sparsity, that is, 
	% the percent of zero values expected, but in order to implemented, it 
	% is necessary to express it as the estimated number of zeros.
	%----------------------------------------------------------------------
	Lam = N*(1-0.01*param.Lam);

	
	% Initializations
	I = diag(ones(1,length(D'*D)));
	S = So;

	Dux = D'*D;
	cS  = max(sqrt(eig(Dux.'*Dux))); % Faster Spectral Norm

	% Constant
	DY = D'*param.data/cS;
	Aq = I-D'*D/cS;

	% --- Main Internal Loop ----------------------------------------------
	for t = 1:Ts

		A = DY + Aq*S;

		% Determine weights
		W = 1./bsxfun(@plus,abs(A),epsilon);

		% Check the KKT conditions
		cnd = (sum(W.*abs(A),2)-Lam')>0;

		% Sources that will potentially require projection
		iS = (1:K);

		% Perform the projection according to the KKT condition
		for i = iS(cnd)
			A(i,:) = DuWProjectOpt(A(i,:),W(i,:),Lam(i));
		end

		% Update
		S = A;
	end
end

%--------------------------------------------------------------------------
%|      function NewCoefUni (So,D,param)                                  |
%|    Calculates the new coefficients imposing sparsity over the whole    |
%| coefficient matrix using the weighted l1-norm projection.              |
%--------------------------------------------------------------------------
function [S] = NewCoefUni(So,D,param)

	% Main parameters
	Ts = param.Ts;     % Internal Loops
	[K,N] = size(So);  % Dimensions

	epsilon = 1e-6;    % Stability control

	%----------------------------------------------------------------------
	%    NOTE: Lam is given by the expected percent of sparsity, that is, 
	% the percent of zero values expected, but in order to implemented, it 
	% is necessary to express it as the estimated number of zeros.
	%----------------------------------------------------------------------
	Lam = N*(1-0.01*param.Lam);


	% Initializations
	I = diag(ones(1,length(D'*D)));
	S = So;

	Dux = D'*D;
	cS  = max(sqrt(eig(Dux.'*Dux))); % Faster Spectral Norm

	% Constant
	DY = D'*param.data/cS;
	Aq = I-D'*D/cS;

	% --- Main Internal Loop ----------------------------------------------
	for t = 1:Ts

		A = DY + Aq*S;

		% Balance the variance among sources
		Var = std(A,0,2);
		A   = bsxfun(@rdivide,A,Var);

		% Determine wights
		W = 1./bsxfun(@plus,abs(A),epsilon);

		% Reshape
		a = reshape(A,1,K*N);
		w = reshape(W,1,K*N);

		% Check the main KKT condition
		cnd = (sum(w.*abs(a))-sum(Lam))>0;

		% Project according to the condition
		if (cnd)
			a = fastWProjectMx(a,w,sum(Lam));

			A = reshape(a,K,N);
		end

		% Return the variance
        A = bsxfun(@times,A,Var);

		% Actualization
		S = A;
	end
end

%--------------------------------------------------------------------------
%|      function NewDict (S,Do,param)                                     |
%|    Calculates the new Dictionary Matrix.                               |
%--------------------------------------------------------------------------
function [D] = NewDict(S,Do,param)

	% Main Parameters
	Td  = param.Td;      % Internal Loops
	ccl = param.ccl;

	% Canonical Dictionary Selection (if it exists)
	if(isfield(param,'Del'))
		Del = param.Del;
		cdl = param.cdl;

		[~,M] = size(Del); % Number of canonical Atoms
	else
		M = 0;  % No Canonical atoms
	end

	% Initializations
	I = diag(ones(1,length(S*S')));
    D = Do;

    Sux = S*S'; 
    cD  = max(sqrt(eig(Sux.'*Sux))); % Faster Spectral Norm

    % Constants
    YS = param.data*S'/cD;
    Bq = I-S*S'/cD;

    % --- Main Internal Loop ----------------------------------------------
    for t = 1:Td

    	% Update B
    	B = YS + D*Bq;

    	% === Constraints ===

 		% Similarity
 		if(M>0)  % If there exits the canonical Dictionary

 			for j = 1:M

 				b = B(:,j);
 				d = Del(:,j);

 				if (norm(b-d)^2 > cdl(j))
 					mu = sqrt(cdl(j))/norm(b-d);
 					B(:,j) = d + mu*(b-d);
 				end
 			end
 		end

 		% Normalization 
 		Kv = sqrt(sum(B.^2));

 		Kv(Kv < ccl) = 1;         % Condition of normalization
        Kv(1:M) = 1;              % Avoid change the Canonical Atoms

        B = bsxfun(@rdivide,B,Kv); % Normalize the rest

        % Update
        D = B;
    end
end

%--------------------------------------------------------------------------
%|      function ReorderMe (Do,So,Lam)                                    |
%|    This function reorders the sources according to Lam depending their |
%| total relative energy.                                                 |
%--------------------------------------------------------------------------
function [D,S,pos] = ReorderMe(Do,So,Lam)

	% Initialization
	D = Do;
	S = So;
	pos = 1:length(Lam);

    [~,iL] = sort(Lam);

    % Normalize size of the sources
    Sk = bsxfun(@rdivide,So,max(abs(So'))');

    % Estimate relative size and order
    wS = sum(abs(Sk),2);

    [~,iS] = sort(wS,'descend');


    % Reorder
    D(:,iL) = D(:,iS);
    S(iL,:) = S(iS,:);

    pos(iL) = pos(iS);

end

%--------------------------------------------------------------------------
%|      function MergeMe (D,S,M)                                          |
%|    This functions studies the correlations between atoms and spatial   |
%| maps, and it merges them if they are similar enough.                   |
%--------------------------------------------------------------------------
function [Dm,Sm,Nmrg] = MergeMe(D,S,M)

	% Parameters
	[T,K] = size(D);
	[~,N] = size(S);

	Nmrg = 0;          % Assume 0 splits

	% Start checking
	for k = M:K

		d = D(:,k);
		s = S(k,:);

		cnt = 1;
		fnd = k;
		i = k+1;

		while(i<=K)
			cpd = D(:,i);
			cps = S(i,:);

			crt = abs(corr(d,cpd));
			crs = abs(corr(s',cps'));

			if(crt>=0.70 || crs>=0.70)
				cnt = cnt + 1;
				fnd(cnt) = i;
			end

			i = i+1;
		end

		if (cnt>1) % A split was found to be collide

			% Found split
			Dc = D(:,fnd);
			Sc = S(fnd,:);

			C = Dc*Sc;

			% Colide the main component using SVD
			[U,Sig,V] = svds(C,1);

			% Add gaussian noise to the sources
            D(:,fnd) = randn(T,cnt);
			S(fnd,:) = randn(cnt,N);

			% Save onlny the principal component 
			D(:,fnd(cnt)) = U;
			S(fnd(cnt),:) = Sig*V';

			clear('U','Sig','V','C','Dc','Sc','fnd');

			Nmrg = Nmrg + 1;

		end
	end

	% Return meged dictionaries

	Dm = D;
	Sm = S;

end


%--------------------------------------------------------------------------
%|                     INITIALIZATION FUNCTION                            |
%+------------------------------------------------------------------------+
%|     This function performs different kind of initialization for MoM    |
%| using several procedures.                                              |
%|     RETURNS                                                            |
%|   The coefficient matrix S and the dictionary D initialized            |
%+------------------------------------------------------------------------+
%|    9-May-2017                                                  -mMm-   |
%--------------------------------------------------------------------------
function [Do,So] = InitiaMe(Opt,Mode)

	%=== Select Mode ===
	switch Mode

		case 'Infomax' % Use Infomax for the full initialization ----------

			% Apply ICA using Infomax
			[S,~] = icaML(Opt.data,Opt.K);

			Do = Opt.data/S;
			So = S;

		case 'Infomax_D' % Use Infomax only for the dictionary ------------

			% Compute ICA full
			[Do,So] = InitiaMe(Opt,'Infomax');

			% Put zeros on the spatial maps
			So = zeros(size(So));


		case 'Infomax_S' % Use Infomax only for the coefficient -----------

			% Compute ICA full
			[Do,So] = InitiaMe(Opt,'Infomax');

			% Put zeros in the Dictionary
			Do = zeros(size(Do));

		case 'Jdr' % Use jadeR for the initialization ---------------------

			% Apply jadeR
			B = jadeR(Opt.data,Opt.K);
			So = B*Opt.data;
			Do = pinv(B);

			% Clear varibles
			clear('B');

		case 'Jdr_D' % Use jadeR only por Dictionary ----------------------

			% Compute jadeR
			[Do,So] = InitiaMe(Opt,'Jdr');

			% Put zeros in the spatial maps
			So = zeros(size(So));

		case 'Jdr_S' % Use jadeR only for the spatial maps ----------------

			% Calculate jadeR
			[Do,So] = InitiaMe(Opt,'Jdr');

			% Put zeros in the dictionary
			Do = zeros(size(Do));

		case 'custom' % Initialization point defined by the user ----------

			Do = Opt.IniD;

			if(isfield(Opt,'IniS'))
				So = Opt.IniS;
			else
				So = pinv(Do)*Opt.data;
			end

		case 'rnd' % Fully Random -----------------------------------------

			% Parameters
			[T,N] = size(Opt.data);
			K     = Opt.K;

			% Random point
			Do = randn(T,K);
			So = randn(K,N);

		otherwise % Error -------------------------------------------------

            error('Intialization mode was not specified!  \(ò-ó)');
	end


	% === Noisy Initial point ===
	if(isfield(Opt,'IniN'))

		% Add random noise to the solution
		Do = Do + Opt.IniN*randn(size(Do));
		So = So + Opt.IniN*randn(size(So));

	end

	% === Normalize data ====
    % Normalization
    nD = sqrt(sum(Do.^2));
    nD(nD < 1) = 1;

    Do = bsxfun(@rdivide,Do,nD);

end