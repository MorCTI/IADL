%-------------------------------------------------------------------------
%|             Matla FAST WEIGHTED L1 NORM PROJECTION                    |
%+-----------------------------------------------------------------------+
%|   This algorithm implement a fast version of the projection over the  |
%| weighted l1-norm ball, given a specific vector, the weights and the   |
%| size of the ball.                                                     |
%+-----------------------------------------------------------------------+
%|   05 Feb 12019                                                 -mMm-  |
%-------------------------------------------------------------------------
function [b] = DuWProjectOpt(a,w,l)

	%--- Parameters ---
	h = abs(a);
	b = h./w;

	sum1 = 0;
	sum2 = 0;

	%=== Main Loop ===
	U = b;
    W = w.*w;

	while(~isempty(U))

		piv = U(end);

		pL = U < U(end);
		pG = U >= U(end);

		L = U(pL);
		G = U(pG);

		lw = W(pL);
		gw = W(pG);

		psum1 = sum(G.*gw);
		psum2 = sum(gw);

		thr = (sum1+psum1-l)/(sum2+psum2);

		% Check pivot
		if(thr<piv)
			sum1 = sum1 + psum1;
			sum2 = sum2 + psum2;

			U = L;
            W = lw;
		else
			U = G(1:end-1);
            W = gw;
		end
	end

	% Set correct threshold
	thr = (sum1-l)/sum2;

	% Apply threshold
	val = h-thr*w;

	b = sign(a).*((val+abs(val))*0.5);
end