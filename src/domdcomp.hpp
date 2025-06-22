/*
 * @file domdcomp.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CANARD_DOMDCOMP_HPP
#define CANARD_DOMDCOMP_HPP

#include <iostream>
#include "common/parameters.hpp"
#include "mpi/check.hpp"

struct domdcomp
{
    domdcomp(int nblocks_) : nblocks(nblocks_)
    {
        check_mpi( MPI_Comm_size(MPI_COMM_WORLD, &mpro) );
        mpro--;

        nbpc[0] = (int *)malloc((nblocks+1)*sizeof(int));
        nbpc[1] = (int *)malloc((nblocks+1)*sizeof(int));
        nbpc[2] = (int *)malloc((nblocks+1)*sizeof(int));
        mo      = (int *)malloc((nblocks+1)*sizeof(int));
        lximb   = (int *)malloc((nblocks+1)*sizeof(int));
        letmb   = (int *)malloc((nblocks+1)*sizeof(int));
        lzemb   = (int *)malloc((nblocks+1)*sizeof(int));
        lxim    = (int *)malloc((mpro+1)*sizeof(int));
        letm    = (int *)malloc((mpro+1)*sizeof(int));
        lzem    = (int *)malloc((mpro+1)*sizeof(int));
        lpos    = (int *)malloc((mpro+1)*sizeof(int));
        nbbc    = (int *)malloc(2*3*(nblocks+1)*sizeof(int));
        mbcd    = (int *)malloc(2*3*(nblocks+1)*sizeof(int));
    }

    ~domdcomp()
    {
    }

    void read_config()
    {
        nbpc[0][0] = 2;
        nbpc[1][0] = 1;
        nbpc[2][0] = 1;
        lximb[0] = 64;
        letmb[0] = 64;
        lzemb[0] = 16;
        for(unsigned int i = 0; i < 6*(nblocks+1); ++i)
        {
            nbbc[i] =  10;
            mbcd[i] = -1;
        }
    }

    void go()
    {
        int myid;

        check_mpi( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );

        mo[0] = 0;
        for (int i=1; i<=nblocks; i++)
            mo[i] = mo[i-1] + nbpc[0][i-1]
                            + nbpc[1][i-1]
                            + nbpc[2][i-1];

        for (int i=0; i<=nblocks; i++) {
            if (myid >= mo[i])
                mb = i;
        }


        lxio = lximb[mb];
        leto = letmb[mb];
        lzeo = lzemb[mb];

        ijkp[0] = (myid - mo[mb])%(nbpc[0][mb]);
        ijkp[1] = (myid - mo[mb] / nbpc[0][mb]) % (nbpc[1][mb]);
        ijkp[2] = (myid - mo[mb] / (nbpc[0][mb] * nbpc[1][mb])) % (nbpc[2][mb]);

        for (int nn = 0; nn < NumberOfSpatialDims; ++nn) {
            int nstart = ((nn+1) % NumberOfSpatialDims);
            int nend   = ((nstart+1) % NumberOfSpatialDims);
            for (int ip = 0; ip < 2; ++ip) {
                int mm = mbcd[mb+nn*(nblocks+1)+ip*3*(nblocks+1)];
                if (mm == -1) {
                    mmcd[ip][nn] = -1;
                    mmcd[ip][nn] = -1;
                } else {
                    mmcd[ip][nn] = idsd3((1 - ip) * (nbpc[nn][mm] - 1),
                                         ijkp[nstart],
                                         ijkp[nend],
                                         mm,
                                         nn);
                }
            }
        }

        for (int nn=0; nn<3; nn++) {
            int ll, mp;
            switch(nn) {
                case 0:
                    ll = lxio;
                    mp = 1;
                    break;
                case 1:
                    ll = leto;
                    mp = nbpc[0][mb];
                    break;
                case 2:
                    ll = lzeo;
                    mp = nbpc[0][mb] * nbpc[1][mb];
                    break;
            }
            int lp = ijkp[nn];
            int ma = nbpc[nn][mb];
            int l;
            if (ma == 1) {
                l = ll;
                nbc[0][nn] = nbbc[mb+nn*(nblocks+1)];
                nbc[1][nn] = nbbc[mb+nn*(nblocks+1)+1*3*(nblocks+1)];
                mcd[0][nn] = mmcd[0][nn];
                mcd[1][nn] = mmcd[1][nn];
            }

            if (ma > 1) {
                if (lp == 0) {
                    l = ll - ( ( ll + 1 ) / ma ) * ( ma - 1 );
                    nbc[0][nn] = nbbc[mb+nn*(nblocks+1)];
                    nbc[1][nn] = BC_INTER_SUBDOMAINS;
                    mcd[0][nn] = mmcd[0][nn];
                    mcd[1][nn] = myid + mp;
                }
                if (lp > 0 && lp < ma-1) {
                    l = ( ll + 1 ) / ma - 1;
                    nbc[0][nn] = BC_INTER_SUBDOMAINS;
                    nbc[1][nn] = BC_INTER_SUBDOMAINS;
                    mcd[0][nn] = myid - mp;
                    mcd[1][nn] = myid + mp;
                }
                if (lp == ma-1) {
                    l = ( ll + 1 ) / ma - 1;
                    nbc[0][nn] = BC_INTER_SUBDOMAINS;
                    nbc[1][nn] = nbbc[mb+nn*(nblocks+1)+3*(nblocks+1)];
                    mcd[0][nn] = myid - mp;
                    mcd[1][nn] = mmcd[1][nn];
                }
            }
            
            switch(nn) {
                case 0:
                    lxi = l;
                    break;
                case 1:
                    let = l;
                    break;
                case 2:
                    lze = l;
                    break;
            }
        }

        if (myid == 0) {
            lxim[0] = lxi;
            letm[0] = let;
            lzem[0] = lze;
            for (int i=1; i<=mpro; i++) {
                int itag = 1;
                check_mpi( MPI_Recv(&lxim[i], 1, MPI_INT, i, itag, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE) );
                itag = 2;
                check_mpi( MPI_Recv(&letm[i], 1, MPI_INT, i, itag, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE) );
                itag = 3;
                check_mpi( MPI_Recv(&lzem[i], 1, MPI_INT, i, itag, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE) );
            }
        } else {
            int itag = 1;
            check_mpi( MPI_Send(&lxi, 1, MPI_INT, 0, itag, MPI_COMM_WORLD) );
            itag = 2;
            check_mpi( MPI_Send(&let, 1, MPI_INT, 0, itag, MPI_COMM_WORLD) );
            itag = 3;
            check_mpi( MPI_Send(&lze, 1, MPI_INT, 0, itag, MPI_COMM_WORLD) );
        }
        check_mpi( MPI_Bcast(lxim, mpro+1, MPI_INT, 0, MPI_COMM_WORLD) );
        check_mpi( MPI_Bcast(letm, mpro+1, MPI_INT, 0, MPI_COMM_WORLD) );
        check_mpi( MPI_Bcast(lzem, mpro+1, MPI_INT, 0, MPI_COMM_WORLD) );
    
        lmx = ( lxi + 1 ) * ( let + 1 ) * ( lze + 1 ) - 1;

        ijk[0][0] = lxi;
        ijk[1][0] = let;
        ijk[2][0] = lze;
        ijk[0][1] = let;
        ijk[1][1] = lze;
        ijk[2][1] = lxi;
        ijk[0][2] = lze;
        ijk[1][2] = lxi;
        ijk[2][2] = let;
        
        nbsize[0] = (ijk[1][0] + 1) * (ijk[2][0] + 1);
        nbsize[1] = (ijk[1][1] + 1) * (ijk[2][1] + 1);
        nbsize[2] = (ijk[1][2] + 1) * (ijk[2][2] + 1);

        for (int i = 0; i <= nblocks; ++i) {
            lpos[mo[i]] = 0;
            for (int j = 1; j < nbpc[0][i]; ++j) {
                int mp = mo[i] + j;
                lpos[mp] = lpos[mp-1] + lxim[mp-1] + 1;
            }
            int jp = nbpc[0][i];
            for (int j = 1; j < nbpc[1][i]; j++) {
                for (int k = 0; k < nbpc[0][i]; k++) {
                    int mp = mo[i] + j * jp + k;
                    lpos[mp] = lpos[mp-jp] + (lximb[i] + 1) * 
                                            (letmb[mp-jp] + 1);
                }
            }
            int kp = nbpc[0][i] * nbpc[1][i];
            for (int j = 1; j < nbpc[2][i]; ++j) {
                for (int k = 0; k < nbpc[1][i]; ++k) {
                    for (int l = 0; l < nbpc[0][i]; ++l) {
                        int mp = mo[i] + j * kp + k * jp + l;
                        lpos[mp] = lpos[mp-kp] + (lximb[i] + 1) *
                                                 (letmb[i] + 1) *
                                                 (lzemb[mp-kp] + 1);
                    }
                }
            }
        }
    }

    void show()
    {
        int myid;

        check_mpi( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );

        std::cout << myid << ": mb        = " << mb  << std::endl;
        std::cout << myid << ": lxi       = " << lxi << std::endl;
        std::cout << myid << ": let       = " << let << std::endl;
        std::cout << myid << ": lze       = " << lze << std::endl;

        std::cout << myid << ": nbc[0][0] = " << nbc[0][0] << std::endl;
        std::cout << myid << ": nbc[0][1] = " << nbc[0][2] << std::endl;
        std::cout << myid << ": nbc[0][2] = " << nbc[0][2] << std::endl;
        std::cout << myid << ": nbc[1][0] = " << nbc[1][0] << std::endl;
        std::cout << myid << ": nbc[1][1] = " << nbc[1][1] << std::endl;
        std::cout << myid << ": nbc[1][2] = " << nbc[1][2] << std::endl;

        std::cout << myid << ": mcd[0][0] = " << mcd[0][0] << std::endl;
        std::cout << myid << ": mcd[0][1] = " << mcd[0][2] << std::endl;
        std::cout << myid << ": mcd[0][2] = " << mcd[0][2] << std::endl;
        std::cout << myid << ": mcd[1][0] = " << mcd[1][0] << std::endl;
        std::cout << myid << ": mcd[1][1] = " << mcd[1][1] << std::endl;
        std::cout << myid << ": mcd[1][2] = " << mcd[1][2] << std::endl;
    }

    int idsd3(int i, int j, int k, int mm, int nn) {
        switch(nn) {
            case 0:
                return mo[mm] + ( k * nbpc[1][mm] + j ) *
                                nbpc[0][mm] + i; 
                break;
            case 1:
                return mo[mm] + ( j * nbpc[1][mm] + i ) *
                                nbpc[0][mm] + k;
                break;
            case 2:
                return mo[mm] + ( i * nbpc[1][mm] + k ) *
                                nbpc[0][mm] + j;
                break;
        }
    }

    int mpro;
    int nblocks;
    int mb;
    int lxio;
    int leto;
    int lzeo;
    int lxi;
    int let;
    int lze;
    int lmx;
    int nbsize[NumberOfSpatialDims];
    int ijkp[NumberOfSpatialDims];
    int ijk[NumberOfSpatialDims][NumberOfSpatialDims];
    int mmcd[2][NumberOfSpatialDims];
    int nbc[2][3];
    int mcd[2][3];
    int *nbpc[NumberOfSpatialDims];
    int *mo;
    int *lximb;
    int *letmb;
    int *lzemb;
    int *lxim;
    int *letm;
    int *lzem;
    int *lpos;
    int *nbbc;
    int *mbcd;
};

#endif
