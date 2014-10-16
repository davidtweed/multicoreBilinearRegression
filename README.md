multicoreBilinearRegression
===========================

Some linear/bilinear regression code (developed for the Azimuth project). Note that the code developed here was written to minimise the program writer's work and make it very easy to be sure the best performance was acheived in the compiled code. As such, it uses techniques such as macros which would be inappropriate in a larger body of software. Hopefully it is still relatively clean though.

**Important notes:** as currently written the program and associated scripts do not pay any real attention to security. Therefore it's incredibly ill-advised to run then in anything other than a standard-user level environment to without making them available to general incoming connections. (In my defence, the vast majority of scientific software is like this, I'm just explicitly mentioning it.) The program attempts to detect errors and shut itself down early rather than let you burn CPU time thinking everything will be OK when it finishes, but beyond that there's no attempt to actually handle errors.

Although I've used the utmost of my abilities in writing this code, it's possible there are bugs. If anyone finds anything worth of comment, I'd be very interested to know; please email david.tweed@gmail.com.
